"""Epoch-based LoRA decision / JSON SFT, with silver-development checkpoint selection.

torch and peft are imported inside main(), so this module imports without a CUDA stack.
"""
import argparse,json,random,time,math,hashlib
from pathlib import Path
from PIL import Image
from gui_verifier.backend import QwenVerifier
from gui_verifier.schema import VerifyRequest
from gui_verifier.prompt import messages_for
from gui_verifier.experiment004 import grouped,json_messages,parse_array,summarize

def main():
 import torch
 from peft import LoraConfig,get_peft_model
 p=argparse.ArgumentParser()
 for arg in ('train','validation','model','output'):p.add_argument('--'+arg,required=True)
 p.add_argument('--mode',choices=['decision','json'],default='decision')
 p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--seed',type=int,default=404)
 p.add_argument('--epochs',type=int,default=3);p.add_argument('--accumulation',type=int,default=8)
 p.add_argument('--micro-batch',type=int,default=4)
 p.add_argument('--max-pixels',type=int,default=786432);p.add_argument('--max-steps',type=int,default=0)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETED').exists():print('Already completed',out);return
 if (out/'train_config.json').exists():raise RuntimeError('Do not overwrite an interrupted training run')
 rng=random.Random(a.seed);random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 train=grouped(a.train);val=grouped(a.validation)
 assert not {r['group_id'] for r in train}&{r['group_id'] for r in val}
 v=QwenVerifier(model=a.model,device='cuda',max_pixels=a.max_pixels)
 targets=[n for n,m in v.model.named_modules() if isinstance(m,torch.nn.Linear) and 'language_model.layers.' in n
  and n.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','out_proj','gate_proj','up_proj','down_proj'}]
 assert targets
 v.model=get_peft_model(v.model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=targets,bias='none'))
 v.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 v.model.enable_input_require_grads();v.model.config.use_cache=False
 params=[p for p in v.model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
 if a.mode=='decision':samples=[(g,i) for g in train for i in range(len(g['gold']))]
 else:
  # Use each supervised judgment once per epoch; natural groups, no duplicated questions.
  samples=[]
  for g in train:
   for start in range(0,len(g['gold']),4):
    samples.append(({**g,'request':{**g['request'],'questions':g['request']['questions'][start:start+4]},
      'gold':g['gold'][start:start+4]},None))
 steps_per_epoch=math.ceil(len(samples)/a.accumulation);total_steps=steps_per_epoch*a.epochs
 if a.max_steps:total_steps=min(total_steps,a.max_steps)
 warmup=max(1,math.ceil(.05*total_steps))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warmup,max(0,(total_steps-s)/max(1,total_steps-warmup))))
 config={**vars(a),'train_decisions':sum(len(g['gold']) for g in train),'training_sequences':len(samples),
  'steps_per_epoch':steps_per_epoch,'planned_steps':total_steps,'warmup':warmup,'target_modules':targets,
  'trainable_parameters':sum(p.numel() for p in params),'rank':16,'alpha':32,'dropout':.05,
  'train_sha256':hashlib.sha256(Path(a.train).read_bytes()).hexdigest(),
  'validation_sha256':hashlib.sha256(Path(a.validation).read_bytes()).hexdigest(),
  'model_revision':v.revision,'selection':'silver validation macro-F1; ties within .002: false-success then earlier',
  'json_batch_note':'same supervised judgment exposure, different sequence count and update budget'}
 (out/'train_config.json').write_text(json.dumps(config,indent=2))
 def prepared_json(g,image):
  return v.processor.apply_chat_template(json_messages(image,g['request']),tokenize=True,
    add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to(v.device)
 def validate():
  records=[];v.model.eval()
  for g in val:
   with Image.open(g['image']) as im:
    image=im.convert('RGB')
    if a.mode=='decision':
     req=VerifyRequest.model_validate(g['request']);scores,_,_=v._score(image,req,req.questions)
     pred=[q.options[max(range(len(s)),key=s.__getitem__)].id for q,s in zip(req.questions,scores)]
    else:
     inp=prepared_json(g,image)
     with torch.inference_mode():
      gen=v.model.generate(**inp,max_new_tokens=16+8*len(g['gold']),do_sample=False,use_cache=True,
       eos_token_id=v.processor.tokenizer.eos_token_id,pad_token_id=v.processor.tokenizer.pad_token_id)
     raw=v.processor.tokenizer.decode(gen[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
     pred=parse_array(raw,len(g['gold']))
    records.append({'sample_id':g['id'],'gold':g['gold'],'predictions':pred})
  v.model.train();return summarize(records),records
 best=None;step=0;started=time.time();torch.cuda.reset_peak_memory_stats()
 with (out/'train_log.jsonl').open('w') as log:
  for epoch in range(a.epochs):
   rng.shuffle(samples);v.model.train()
   for offset in range(0,len(samples),a.accumulation):
    batch=samples[offset:offset+a.accumulation];optimizer.zero_grad(set_to_none=True);loss_value=0
    if a.mode=='decision':
     for micro_start in range(0,len(batch),a.micro_batch):
      micro=batch[micro_start:micro_start+a.micro_batch];conversations=[];targets_gold=[]
      for g,i in micro:
       req=VerifyRequest.model_validate(g['request']);q=req.questions[i].model_copy(deep=True);rng.shuffle(q.options)
       assert len(q.options)==3,'Experiment 004 is fixed yes/no/unknown'
       targets_gold.append(next(j for j,o in enumerate(q.options) if o.id==g['gold'][i]))
       with Image.open(g['image']) as im:conversations.append(messages_for(im.convert('RGB'),req,q))
      inp=v.processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,
       enable_thinking=False,return_dict=True,return_tensors='pt',padding=True).to(v.device)
      with torch.autocast('cuda',dtype=torch.bfloat16):
       result=v.model(**inp,use_cache=False,logits_to_keep=1)
       logits=result.logits[:,-1,v.label_ids[:3]].float()
       loss=torch.nn.functional.cross_entropy(logits,torch.tensor(targets_gold,device='cuda'))
      if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
      weight=len(micro)/len(batch);(loss*weight).backward();loss_value+=loss.item()*weight
    for g,i in ([] if a.mode=='decision' else batch):
     order=list(range(len(g['gold'])));rng.shuffle(order)
     g={**g,'gold':[g['gold'][j] for j in order],
        'request':{**g['request'],'questions':[g['request']['questions'][j] for j in order]}}
     with Image.open(g['image']) as im:
      image=im.convert('RGB')
      if a.mode=='decision':
       req=VerifyRequest.model_validate(g['request']);q=req.questions[i].model_copy(deep=True);rng.shuffle(q.options)
       gold=g['gold'][i];target=next(j for j,o in enumerate(q.options) if o.id==gold)
       inp=v.prepare(image,req,[q])
       with torch.autocast('cuda',dtype=torch.bfloat16):
        result=v.model(**inp,use_cache=False,logits_to_keep=1)
        logits=result.logits[:,-1,v.label_ids[:len(q.options)]].float()
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([target],device='cuda'))
      else:
       inp=prepared_json(g,image)
       answer=json.dumps(g['gold'],separators=(',',':'))
       ids=v.processor.tokenizer.encode(answer,add_special_tokens=False)+[v.processor.tokenizer.eos_token_id]
       target=torch.tensor([ids],device='cuda');n=len(ids)
       inp['input_ids']=torch.cat([inp['input_ids'],target],dim=1)
       inp['attention_mask']=torch.cat([inp['attention_mask'],torch.ones_like(target)],dim=1)
       if 'mm_token_type_ids' in inp:inp['mm_token_type_ids']=torch.cat([inp['mm_token_type_ids'],torch.zeros_like(target)],dim=1)
       with torch.autocast('cuda',dtype=torch.bfloat16):
        result=v.model(**inp,use_cache=False,logits_to_keep=n+1)
        loss=torch.nn.functional.cross_entropy(result.logits[:,:-1,:].float().reshape(-1,result.logits.shape[-1]),target.reshape(-1))
      if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
      (loss/len(batch)).backward();loss_value+=loss.item()/len(batch)
    norm=torch.nn.utils.clip_grad_norm_(params,1.0)
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient norm')
    optimizer.step();scheduler.step();step+=1
    entry={'step':step,'epoch':epoch+(offset+len(batch))/len(samples),'loss':loss_value,
     'lr':scheduler.get_last_lr()[0],'grad_norm':float(norm),'elapsed_s':time.time()-started,
     'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    log.write(json.dumps(entry)+'\n');log.flush()
    if step%10==0 or step==1:print(json.dumps(entry),flush=True)
    boundary=(offset//a.accumulation+1) in {math.ceil(steps_per_epoch/2),steps_per_epoch} or step==total_steps
    if boundary:
     metrics,records=validate();metrics.update(step=step,epoch=entry['epoch'])
     (out/f'validation-step-{step}.json').write_text(json.dumps({'metrics':metrics,'records':records}))
     print(json.dumps({'validation':metrics}),flush=True)
     better=best is None or metrics['macro_f1']>best['macro_f1']+.002 or (
       abs(metrics['macro_f1']-best['macro_f1'])<=.002 and metrics['false_success_rate']<best['false_success_rate'])
     if better:
      best=metrics;v.model.save_pretrained(out/'best-adapter');v.processor.save_pretrained(out/'best-adapter')
      (out/'best.json').write_text(json.dumps(best,indent=2))
     # Latest checkpoint enables recovery of a long-running job without discarding weights.
     v.model.save_pretrained(out/'latest-adapter')
     torch.save({'step':step,'epoch':epoch,'offset_next':offset+len(batch),'optimizer':optimizer.state_dict(),
       'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
       'python_rng':rng.getstate()},out/'optimizer-latest.pt')
    if step>=total_steps:break
   if step>=total_steps:break
 (out/'COMPLETED').write_text(json.dumps({'steps':step,'best':best,'elapsed_s':time.time()-started}))
if __name__=='__main__':main()
