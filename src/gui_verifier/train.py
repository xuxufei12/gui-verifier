"""BF16 LoRA with candidate-restricted CE at the final prompt position.

No reasoning/JSON targets; no gold label is included in model input.
torch and peft are imported inside main(), so the CPU-only test suite can collect
this module and reuse assert_disjoint without a CUDA stack installed.
"""
import argparse
import json
import random
import time
from pathlib import Path

from PIL import Image

from .backend import QwenVerifier
from .schema import VerifyRequest


def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def assert_disjoint(train, validation):
    overlap={r['group_id'] for r in train}&{r['group_id'] for r in validation}
    if overlap: raise ValueError(f'Train/validation group leakage: {len(overlap)} groups')


def main():
    import torch
    from peft import LoraConfig, get_peft_model

    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--model',default='Qwen/Qwen3.5-0.8B');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=48);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--rank',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=20260922);p.add_argument('--max-pixels',type=int,default=512*512)
    p.add_argument('--label-smoothing',type=float,default=0.0)
    p.add_argument('--save-every',type=int,default=0)
    a=p.parse_args()
    if a.steps<1 or a.accumulation<1: raise ValueError('Positive steps and accumulation required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This BF16 training recipe requires a BF16-capable CUDA GPU')
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    rng=random.Random(a.seed)
    train=load_rows(a.train);validation=load_rows(a.validation);assert_disjoint(train,validation)
    samples=[(r,q['id']) for r in train for q in r['request']['questions']]
    if not samples: raise ValueError('Empty training set')
    rng.shuffle(samples)
    v=QwenVerifier(model=a.model,device='cuda',max_pixels=a.max_pixels)
    # Restrict adapters to language layers; include hybrid DeltaNet projections.
    targets=[name for name,module in v.model.named_modules()
             if isinstance(module,torch.nn.Linear) and 'language_model.layers.' in name
             and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','out_proj','gate_proj','up_proj','down_proj'}]
    if not targets: raise RuntimeError('No matching language LoRA projections found')
    v.model=get_peft_model(v.model,LoraConfig(r=a.rank,lora_alpha=2*a.rank,lora_dropout=0.05,
                                            target_modules=targets,bias='none'))
    v.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    v.model.enable_input_require_grads()
    v.model.config.use_cache=False
    v.model.train()
    # The frozen vision tower must not retain a training graph.
    for name,param in v.model.named_parameters():
        if '.visual.' in name: param.requires_grad_(False)
    params=[p for p in v.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.01)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    config={**vars(a),'trainable_parameters':sum(p.numel() for p in params),'target_modules':targets,
            'loss':'candidate-restricted cross entropy; option order shuffled each presentation',
            'train_samples':len(samples),'validation_used_for':'group-leakage check; external evaluation after training',
            'model_revision':v.revision}
    (out/'train_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    cursor=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/'train_log.jsonl').open('w',encoding='utf-8') as log:
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True);loss_sum=0.0
            for micro in range(a.accumulation):
                if cursor==len(samples): rng.shuffle(samples);cursor=0
                row,qid=samples[cursor];cursor+=1
                req=VerifyRequest.model_validate(row['request'])
                q=next(q for q in req.questions if q.id==qid).model_copy(deep=True)
                rng.shuffle(q.options)
                gold=row['labels'][qid];target=next(i for i,o in enumerate(q.options) if o.id==gold)
                with Image.open(Path(a.train).parent/row['image']) as im:
                    inputs=v.prepare(im.convert('RGB'),req,[q])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=v.model(**inputs,use_cache=False,logits_to_keep=1)
                    logits=output.logits[:,-1,v.label_ids[:len(q.options)]].float()
                    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([target],device='cuda'),label_smoothing=a.label_smoothing)
                (loss/a.accumulation).backward();loss_sum+=loss.detach().item()/a.accumulation
            norm=torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step()
            entry={'step':step+1,'loss':loss_sum,'grad_norm':float(norm),
                   'elapsed_s':time.perf_counter()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            print(json.dumps(entry),flush=True);log.write(json.dumps(entry)+'\n');log.flush()
            if (step+1)%16==0: v.model.save_pretrained(out/'latest')
            if a.save_every and (step+1)%a.save_every==0:v.model.save_pretrained(out/f'checkpoint-{step+1}')
        v.model.save_pretrained(out/'adapter')
        v.processor.save_pretrained(out/'adapter')
    (out/'completed.json').write_text(json.dumps(entry,indent=2),encoding='utf-8')

if __name__=='__main__':main()
