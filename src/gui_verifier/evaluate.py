import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image
from .backend import QwenVerifier
from .schema import VerifyRequest


def metrics(records):
    if not records: raise ValueError('No decisions to evaluate')
    n=len(records)
    accuracy=sum(r['correct'] for r in records)/n
    brier=sum(sum((p-(k==r['gold']))**2 for k,p in r['probabilities'].items()) for r in records)/n
    nll=-sum(math.log(max(r['probabilities'][r['gold']],1e-12)) for r in records)/n
    ece=0
    for b in range(10):
        bucket=[r for r in records if min(int(r['max_probability']*10),9)==b]
        if bucket:
            ece+=len(bucket)/n*abs(sum(r['correct'] for r in bucket)/len(bucket)-sum(r['max_probability'] for r in bucket)/len(bucket))
    accepted=[r for r in records if r['status']=='decided']
    order=[r for r in records if r.get('order_consistent') is not None]
    classes=sorted({r['gold'] for r in records}|{r.get('selected_option_id', '') for r in records})
    classes=[c for c in classes if c]
    f1=[]
    confusion={gold:{pred:0 for pred in classes} for gold in classes}
    for r in records:
        pred=r.get('selected_option_id',max(r['probabilities'],key=r['probabilities'].get))
        if pred in classes: confusion[r['gold']][pred]+=1
    for label in classes:
        tp=confusion[label][label]
        fp=sum(confusion[k][label] for k in classes if k!=label)
        fn=sum(confusion[label][k] for k in classes if k!=label)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {'n':n,'accuracy':accuracy,'multiclass_brier_sum':brier,'nll':nll,'ece_10_equal_width':ece,
            'macro_f1':sum(f1)/len(f1),'confusion_gold_by_pred':confusion,
            'option_order_flip_rate':sum(not r['order_consistent'] for r in order)/len(order) if order else None,
            'coverage_decided':len(accepted)/n,
            'error_rate_decided':1-sum(r['correct'] for r in accepted)/len(accepted) if accepted else None,
            'unknown_prediction_rate':sum(r['status']=='unknown' for r in records)/n}


def run(verifier, data, output, check_order=False, limit=None):
    data=Path(data); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in data.read_text(encoding='utf-8').splitlines() if line.strip()]
    if limit: rows=rows[:limit]
    records=[]; times=[]
    verifier.torch.cuda.reset_peak_memory_stats() if verifier.device.startswith('cuda') else None
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            req=VerifyRequest.model_validate(row['request']); req.check_order=check_order
            with Image.open(data.parent/row['image']) as im:
                result=verifier.verify(im.convert('RGB'),req)
            times.append(result.timing_ms['total'])
            for d in result.decisions:
                record={'sample_id':row['id'],'group_id':row['group_id'], 'gold':row['labels'][d.question_id],**d.model_dump()}
                record['correct']=record['gold']==record['selected_option_id']
                records.append(record)
                f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()
    groups=defaultdict(list)
    for r in records: groups[r['question_id']].append(r)
    summary={'data':str(data),'model':verifier.model_name,'adapter':verifier.adapter,
             'metrics':metrics(records),'by_question':{k:metrics(v) for k,v in groups.items()},
             'latency_ms':{'median_request':statistics.median(times),'max_request':max(times),
                           'first_request':times[0],'includes_preprocess':True,'includes_model_load':False},
             'peak_allocated_gib':verifier.torch.cuda.max_memory_allocated()/2**30 if verifier.device.startswith('cuda') else None,
             'metadata':result.metadata}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--model',default='Qwen/Qwen3.5-0.8B');p.add_argument('--adapter');p.add_argument('--limit',type=int)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--max-pixels',type=int,default=512*512)
    p.add_argument('--fast',action='store_true')
    p.add_argument('--check-order',action='store_true');a=p.parse_args()
    if a.fast:
        from .fast_backend import FastQwenVerifier
        cls=FastQwenVerifier
    else:cls=QwenVerifier
    v=cls(model=a.model,adapter=a.adapter,batch_size=a.batch_size,max_pixels=a.max_pixels)
    run(v,a.data,a.output,a.check_order,a.limit)

if __name__=='__main__':main()
