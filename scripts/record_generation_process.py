"""Observe actual final AR/Jev forward outputs without changing sampling decisions."""
import argparse
import json
from pathlib import Path
import torch
from djev import schema
from djev.infill import ar_decide, denoise_infill, readout
from djev.io import load
from djev.sampler import SamplerConfig
from djev.sft import parse_questions

REQUEST={'state':'Support ticket: My invoice was charged twice. I need the duplicate payment refunded today. The service is working normally. I do not want to cancel my subscription. Please contact me by email.',
 'questions':[{'type':'noul','id':id,'question':q} for id,q in [
 ('q1','Does the customer report a duplicate charge?'),('q2','Does the customer request a refund?'),
 ('q3','Does the customer request action today?'),('q4','Is the service currently unavailable?'),
 ('q5','Does the customer want to cancel the subscription?'),('q6','Does the customer request contact by email?'),
 ('q7','Does the issue concern billing?'),('q8','Does the customer request a delivery address change?')]],
 'display_labels':['二重請求','返金要求','今日の対応','サービス停止','解約希望','メール連絡','請求の問題','住所変更']}

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['ar','diffusion'],required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 torch.manual_seed(20261025)
 ck='runs/jev_infill' if args.mode=='diffusion' else 'runs/ar_sft_v5'
 model,tok,proc=load('Qwen/Qwen3-VL-4B-Instruct',ck)
 qs=parse_questions(REQUEST['questions']);template,slots,candidates=schema.infill_template(tok,REQUEST['state'],qs)
 assert len(template)<=128
 records=[]
 if args.mode=='diffusion':
  ctx=model.encode(torch.tensor([schema.prompt_ids(tok,REQUEST['state'],qs)],device=model.device))
  original=model.decode
  def observe(x,z,context):
   logits=original(x,z,context)
   probs=[readout(logits[0,pos],cand) for pos,cand in zip(slots,candidates)]
   records.append({'step':len(records)+1,'input_ids':x[0,slots].tolist(),
    'input_tokens':[tok.decode([i]) for i in x[0,slots].tolist()],
    'unrestricted_predictions':[tok.decode([int(logits[0,pos].argmax())]) for pos in slots],
    'predictions':[max(p,key=p.get) for p in probs],'candidate_probs':probs})
   return logits
  model.decode=observe
  _,first,final,steps=denoise_infill(model,ctx,template,slots,SamplerConfig(canvas_length=128,max_steps=48))
  model.decode=original
  actual=[max(readout(l,c),key=readout(l,c).get) for l,c in zip(first,candidates)]
  assert actual==records[0]['predictions'] and steps==len(records)
 else:
  counter=[0]
  def observe(module,unused,output):
   i=counter[0];p=readout(output.logits[0,-1],candidates[i]);answer=max(p,key=p.get)
   records.append({'step':i+1,'question_id':qs[i].id,'prediction':answer,'candidate_probs':p});counter[0]+=1
  hook=model.backbone.register_forward_hook(observe)
  try:actual_dict,steps=ar_decide(model.backbone,tok,REQUEST['state'],qs)
  finally:hook.remove()
  actual=[actual_dict[q.id]['answer'] for q in qs]
  assert actual==[s['prediction'] for s in records] and steps==len(records)
 result={'mode':args.mode,'checkpoint':ck,'seed':20261025,'input':REQUEST,'answer_template':template,'slots':slots,'steps':records,'api_answers':actual,
 'display_note':'Diffusion cards show candidate-restricted argmax at each decode, not the raw noisy canvas or irreversible acceptance. API reads first. AR uses one forward per blank, not one forward per JSON character. Animation step rate is illustrative, not elapsed-time aligned.'}
 Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({'mode':args.mode,'forwards':steps,'api_answers':actual}),flush=True)

if __name__=='__main__':main()
