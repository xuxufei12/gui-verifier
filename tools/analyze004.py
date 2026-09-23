"""Paired group bootstrap and calibration; never fits temperature on validation/test."""
import argparse,json,math,random,statistics
from pathlib import Path
from collections import defaultdict
from gui_verifier.experiment004 import summarize,LABELS

def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def interval(values):
 values=sorted(values);return [values[int(.025*len(values))],values[min(len(values)-1,int(.975*len(values)))]]
def pair_bootstrap(left,right,reps=2000):
 l={r['sample_id']:r for r in left};r={r['sample_id']:r for r in right};assert l.keys()==r.keys()
 groups=defaultdict(list)
 for sid,row in l.items():
  assert row['gold']==r[sid]['gold'];assert row['group_id']==r[sid]['group_id']
  groups[row['group_id']].append(sid)
 keys=list(groups);rng=random.Random(404);values=defaultdict(list)
 for _ in range(reps):
  ids=[sid for g in rng.choices(keys,k=len(keys)) for sid in groups[g]]
  a=summarize([l[i] for i in ids]);b=summarize([r[i] for i in ids])
  for k in ('macro_f1','strict_accuracy','false_success_rate'):values[k].append(b[k]-a[k])
 return {'group_count':len(keys),'bootstrap_replicates':reps,'delta_right_minus_left_95CI':{k:interval(v) for k,v in values.items()}}
def calibrated(probs,t):
 logits=[math.log(max(p,1e-12))/t for p in probs];m=max(logits);e=[math.exp(x-m) for x in logits];return [x/sum(e) for x in e]
def probability_metrics(rows,t):
 entries=[(calibrated(p,t),LABELS.index(g)) for r in rows for p,g in zip(r['probabilities'],r['gold'])]
 n=len(entries);nll=sum(-math.log(max(p[g],1e-12)) for p,g in entries)/n
 brier=sum(sum((v-(i==g))**2 for i,v in enumerate(p)) for p,g in entries)/n
 bins=defaultdict(list)
 for p,g in entries:bins[min(9,int(max(p)*10))].append((max(p),p.index(max(p))==g))
 ece=sum(len(b)/n*abs(sum(x for x,_ in b)/len(b)-sum(y for _,y in b)/len(b)) for b in bins.values())
 return {'n':n,'nll':nll,'brier_sum':brier,'ece_10':ece}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
 runs={n:read(root/n/'predictions.jsonl') for n in ('base-full','decision-selected-full','json-selected-full')}
 get=lambda n,m:[r for r in runs[n] if r['method']==m]
 comparisons={}
 for name,la,ra in [('training',('base-full','fast'),('decision-selected-full','fast')),
                   ('base_protocol',('base-full','json'),('base-full','fast')),
                   ('base_constrained_protocol',('base-full','json_constrained'),('base-full','fast')),
                   ('trained_protocol',('decision-selected-full','json'),('decision-selected-full','fast')),
                   ('protocol_adapted_systems',('json-selected-full','json'),('decision-selected-full','fast'))]:
  comparisons[name]=pair_bootstrap(get(*la),get(*ra))
 cal=[r for r in read(root/'calibration-eval/predictions.jsonl') if r['method']=='fast']
 temps=[math.exp(math.log(.25)+i/80*math.log(16)) for i in range(81)]
 t=min(temps,key=lambda x:probability_metrics(cal,x)['nll'])
 validation=get('decision-selected-full','fast')
 calibration={'temperature':t,'fit_split':'calibration only','calibration':probability_metrics(cal,t),
  'silver_validation_before':probability_metrics(validation,1),'silver_validation_after':probability_metrics(validation,t)}
 speed={}
 for run in ('base-full','decision-selected-full','json-selected-full'):
  rows=read(root/run/'timings.jsonl');speed[run]={}
  for k in (1,2,4,8):
   by=defaultdict(lambda:defaultdict(list));groups={}
   for r in rows:
    if r['k']==k:by[r['sample_id']][r['method']].append(r['latency_ms']);groups[r['sample_id']]=r['group_id']
   cluster=defaultdict(list)
   for sid,m in by.items():
    if 'json' in m and 'fast' in m:cluster[groups[sid]].append((statistics.median(m['json']),statistics.median(m['fast'])))
   if not cluster:speed[run][str(k)]={'available':False};continue
   pairs=[p for values in cluster.values() for p in values];rng=random.Random(404);ratios=[];keys=list(cluster)
   for _ in range(2000):
    sample=[p for g in rng.choices(keys,k=len(keys)) for p in cluster[g]]
    ratios.append(statistics.median(x for x,y in sample)/statistics.median(y for x,y in sample))
   speed[run][str(k)]={'ratio_of_screen_medians':statistics.median(x for x,y in pairs)/statistics.median(y for x,y in pairs),
    'median_paired_speedup':statistics.median(x/y for x,y in pairs),'ratio_95CI':interval(ratios),'screens':len(pairs)}
 report={'comparisons':comparisons,'speed':speed,'calibration':calibration,
  'limitations':['silver development labels, not independent human test','hyperparameters selected on this validation split',
  'bootstrap intervals conditional on selected weights; not a guarantee of generalization','K8 not fabricated']}
 (root/'analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
