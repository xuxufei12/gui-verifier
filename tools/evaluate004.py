"""Full-array generation and direct scoring, same weights and image budget."""
import argparse,json,time,random,statistics,math,hashlib,itertools
from collections import defaultdict
from pathlib import Path
from PIL import Image
from gui_verifier.experiment004 import grouped,json_messages,parse_array,subset,summarize
from gui_verifier.schema import VerifyRequest
from gui_verifier.fast_backend import FastQwenVerifier
from gui_verifier.backend import QwenVerifier

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',required=True)
 p.add_argument('--adapter');p.add_argument('--output',required=True);p.add_argument('--limit',type=int)
 p.add_argument('--max-pixels',type=int,default=786432);p.add_argument('--speed-screens',type=int,default=40)
 p.add_argument('--repeats',type=int,default=5);p.add_argument('--methods',default='json,json_constrained,logits,fast')
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETED').exists():print('Already completed',out);return
 if (out/'predictions.jsonl').exists():raise RuntimeError('Incomplete output exists; use a fresh directory')
 v=QwenVerifier(model=a.model,adapter=a.adapter,max_pixels=a.max_pixels,batch_size=4)
 torch=v.torch;v.model.eval();rows=grouped(a.data);random.Random(404).shuffle(rows)
 if a.limit:rows=rows[:a.limit]
 merge_check=None
 if a.adapter:
  g=rows[0];req=VerifyRequest.model_validate(g['request'])
  with Image.open(g['image']) as im:
   image=im.convert('RGB');before,_,_=v._score(image,req,req.questions)
   v.model=v.model.merge_and_unload();v.model.eval()
   after,_,_=v._score(image,req,req.questions)
  merge_check={'max_abs_logit_delta':max(abs(x-y) for b,c in zip(before,after) for x,y in zip(b,c)),
   'prediction_agreement':all(max(range(len(b)),key=b.__getitem__)==max(range(len(c)),key=c.__getitem__) for b,c in zip(before,after)),
   'checked_screens':1,'scope':'one-screen smoke, not exhaustive merge equivalence'}
 methods=a.methods.split(',');assert set(methods)<= {'json','json_constrained','logits','fast'}
 grammar={}
 if 'json_constrained' in methods:
  grammar_start=time.perf_counter()
  for k in sorted({len(g['gold']) for g in rows}|{k for k in (1,2,4,8) if any(len(g['gold'])>=k for g in rows)}):
   assert k<=8,'Finite JSON grammar limited to 8 questions'
   trie={}
   for labels in itertools.product(('yes','no','unknown'),repeat=k):
    ids=v.processor.tokenizer.encode(json.dumps(labels,separators=(',',':')),add_special_tokens=False)
    node=trie
    for token in ids+[v.processor.tokenizer.eos_token_id]:node=node.setdefault(token,{})
   grammar[k]=trie
  grammar_ms=(time.perf_counter()-grammar_start)*1000
 else:grammar_ms=0
 cfg={**vars(a),'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
  'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'model_revision':v.revision,
  'timing':'decoded RGB through output array, synchronized; excludes disk/model load',
  'adapter_merged':bool(a.adapter),'merge_check':merge_check,'shared_text_prefix':False,
  'grammar_build_ms':grammar_ms,'grammar_cache':'label-array schema only, no request/image/answer cache',
  'generation_eos_token_id':v.processor.tokenizer.eos_token_id,'generation_pad_token_id':v.processor.tokenizer.pad_token_id,
  'quality':'silver development evaluation'}
 (out/'manifest.json').write_text(json.dumps(cfg,indent=2))
 def run(g,image,method):
  req=VerifyRequest.model_validate(g['request']);v._sync();start=time.perf_counter()
  raw=None;n_tokens=0;probs=None
  if method in ('json','json_constrained'):
   inp=v.processor.apply_chat_template(json_messages(image,g['request']),tokenize=True,
     add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to(v.device)
   # Conservative per-label bound, includes punctuation, whitespace and EOS margin.
   k=len(g['gold']);bound=16+k*(max(len(v.processor.tokenizer.encode(json.dumps(x),add_special_tokens=False)) for x in ('yes','no','unknown'))+4)
   kwargs={}
   if method=='json_constrained':
    prompt_length=inp['input_ids'].shape[1]
    def allowed(batch_id,sequence):
     node=grammar[k]
     for token in sequence[prompt_length:].tolist():node=node[token]
     return list(node)
    kwargs['prefix_allowed_tokens_fn']=allowed
   with torch.inference_mode():
    gen=v.model.generate(**inp,max_new_tokens=bound,do_sample=False,use_cache=True,
      eos_token_id=v.processor.tokenizer.eos_token_id,pad_token_id=v.processor.tokenizer.pad_token_id,**kwargs)
   tokens=gen[0,inp['input_ids'].shape[1]:];n_tokens=len(tokens)
   raw=v.processor.tokenizer.decode(tokens,skip_special_tokens=True);pred=parse_array(raw,k)
   input_tokens=int(inp['attention_mask'].sum())
  else:
   # One independent language row per question for both L and F.
   scorer=QwenVerifier._score if method=='logits' else FastQwenVerifier._score
   scores,_,_=scorer(v,image,req,req.questions)
   probs=[torch.tensor(s).softmax(-1).tolist() for s in scores]
   pred=[q.options[max(range(len(s)),key=s.__getitem__)].id for q,s in zip(req.questions,scores)]
   input_tokens=None
  encoded=json.dumps(pred,separators=(',',':')) if pred is not None else None
  v._sync();elapsed=(time.perf_counter()-start)*1000
  return {'sample_id':g['id'],'group_id':g['group_id'],'method':method,'k':len(g['gold']),
    'gold':g['gold'],'predictions':pred,'raw_output':raw,'output_json':encoded,
    'probabilities':probs,'latency_ms':elapsed,'output_tokens':n_tokens,'input_tokens':input_tokens,
    'original_ids':g['original_ids']}
 # Warm all methods; no disk loading inside timers.
 for g in rows[:min(10,len(rows))]:
  with Image.open(g['image']) as im:
   image=im.convert('RGB')
   for method in methods:run(g,image,method)
 records=[];torch.cuda.reset_peak_memory_stats()
 with (out/'predictions.jsonl').open('w') as f:
  for i,g in enumerate(rows):
   with Image.open(g['image']) as im:
    image=im.convert('RGB');order=methods[:];random.Random(404+i).shuffle(order)
    for method in order:
     r=run(g,image,method);records.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
   if (i+1)%10==0:print(json.dumps({'evaluated_screens':i+1,'total':len(rows)}),flush=True)
 summary={m:summarize([r for r in records if r['method']==m]) for m in methods}
 equivalence=[]
 if 'logits' in methods and 'fast' in methods:
  left={r['sample_id']:r for r in records if r['method']=='logits'}
  for r in records:
   if r['method']=='fast':
    other=left[r['sample_id']]
    equivalence.append({'sample_id':r['sample_id'],'agreement':r['predictions']==other['predictions'],
      'max_probability_delta':max(abs(x-y) for a,b in zip(r['probabilities'],other['probabilities']) for x,y in zip(a,b))})
 # Require natural distinct questions; report missing K=8 rather than fabricate repeats.
 timed=[]
 with (out/'timings.jsonl').open('w') as f:
  for k in (1,2,4,8):
   strata=defaultdict(list)
   for g in rows:
    if len(g['gold'])<k:continue
    filename=Path(g['image']).name
    domain='mobile' if filename.startswith('screenqa') else ('desktop' if 'desktop' in filename else 'web')
    with Image.open(g['image']) as im:area=im.width*im.height
    strata[(domain,area>1000000,'unknown' in g['gold'])].append(g)
   eligible=[]
   while len(eligible)<a.speed_screens and any(strata.values()):
    for key in sorted(strata):
     if strata[key] and len(eligible)<a.speed_screens:eligible.append(strata[key].pop(0))
   for g0 in eligible:
    g=subset(g0,k)
    with Image.open(g['image']) as im:
     image=im.convert('RGB')
     for method in methods:run(g,image,method)
     for rep in range(a.repeats):
      order=methods[:];random.Random(g['id']+str(rep)).shuffle(order)
      for method in order:
       r=run(g,image,method);r['repeat']=rep;timed.append(r);f.write(json.dumps(r)+'\n');f.flush()
   print(json.dumps({'timing_k':k,'screens':len(eligible)}),flush=True)
 speed={}
 for k in (1,2,4,8):
  speed[str(k)]={}
  for m in methods:
   rs=[r for r in timed if r['k']==k and r['method']==m];ts=sorted(r['latency_ms'] for r in rs)
   speed[str(k)][m]={'screens':len({r['sample_id'] for r in rs}),'measurements':len(ts),
     'p50_ms':statistics.median(ts) if ts else None,'p95_ms':ts[math.ceil(.95*len(ts))-1] if ts else None,
     'attempted_decisions_per_second':len(ts)*k*1000/sum(ts) if ts else None,
     'valid_decisions_per_second':sum(r['predictions'] is not None for r in rs)*k*1000/sum(ts) if ts else None,
     'valid_array_rate':sum(r['predictions'] is not None for r in rs)/len(rs) if rs else None}
 report={'quality':summary,'speed':speed,'fast_equivalence':equivalence,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
  'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
  'limitations':['No independent human gold','K=8 unavailable when fewer than 8 unique questions per screenshot',
   'JSON sees all questions; logits uses independent question prompts','No confidence intervals yet; raw paired records retained']}
 (out/'summary.json').write_text(json.dumps(report,indent=2));(out/'COMPLETED').write_text('ok\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
