"""Render recorded final AR/Jev forwards as actual candidate prediction cards."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

FONT='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
BOLD='/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'
def font(n,bold=False):return ImageFont.truetype(BOLD if bold else FONT,n)
BG='#F6F1E8';INK='#382F29';MUTED='#796F64';GREEN='#617447';ORANGE='#B66235'
W,H=1320,710;PW,PH=612,480;Y=138;XS=[32,676]

def panel(record,k,mode):
 im=Image.new('RGB',(PW,PH), '#EEE5D5' if mode=='ar' else '#F3DFD1');d=ImageDraw.Draw(im)
 steps=record['steps'];n=min(k,len(steps));color=GREEN if mode=='ar' else ORANGE
 head=f'AR：回答 {n}/{len(steps)}' if mode=='ar' else f'拡散：反復 {n}/{len(steps)}'
 d.text((18,10),head,font=font(28,True),fill=color)
 note=('1つずつ回答を追加' if mode=='ar' else ('全穴に乱数を入力' if not k else '全8項目の予測を同時更新'))
 d.text((265,16),note,font=font(22),fill=INK)
 for i,label in enumerate(record['input']['display_labels']):
  x=16+(i%2)*298;y=64+(i//2)*100;changed=False;value=None;p=None
  if mode=='ar' and i<n:
   s=steps[i];value=s['prediction'];p=s['candidate_probs'][value];changed=(i==n-1 and k<=len(steps))
  elif mode=='diffusion' and n:
   s=steps[n-1];value=s['predictions'][i];p=s['candidate_probs'][i][value]
   changed=k<=len(steps) and (n==1 or value!=steps[n-2]['predictions'][i])
  fill='#F9DC9A' if changed else '#FFFCF7'
  d.rounded_rectangle((x,y,x+282,y+90),radius=12,fill=fill)
  d.text((x+12,y+5),label,font=font(23,True),fill=INK)
  answer=('はい' if value=='true' else 'いいえ') if value else ('未判定' if mode=='ar' else '乱数入力')
  raw = record['steps'][min(max(k-1,0),len(steps)-1)]['input_tokens'][i] if mode=='diffusion' else None
  if mode=='diffusion':
   safe=raw.encode('unicode_escape').decode('ascii')[:13]
   d.text((x+12,y+34),'入力：'+safe,font=font(17),fill=MUTED)
  d.text((x+12,y+56 if mode=='diffusion' else y+38),answer,font=font(24 if mode=='diffusion' else 29,True),fill=color if value else MUTED)
  if p is not None:
   d.text((x+171,y+62 if mode=='diffusion' else y+48),f'{p*100:.1f}%',font=font(19),fill=MUTED)
   d.rectangle((x+12,y+82,x+12+int(258*p),y+86),fill=color)
 if mode=='diffusion':
  note='初回の予測をAPIが返す' if n==1 else ('途中の候補予測。確定ではない' if n else '最初から全ての穴を対象にする')
 else:note='既に答えた項目を使って次へ進む'
 d.text((18,458),note,font=font(17),fill=MUTED)
 return im

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ar',required=True);ap.add_argument('--diffusion',required=True);ap.add_argument('--out-dir',default='docs/generation');a=ap.parse_args()
 ar=json.load(open(a.ar));df=json.load(open(a.diffusion));assert ar['input']==df['input'];n=max(len(ar['steps']),len(df['steps']));frames=[];ars=[];dfs=[]
 for k in range(n+1):
  im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
  d.text((32,16),'最終ARと拡散Jev：同じ8問への回答過程',font=font(33,True),fill=INK)
  d.text((32,65),'入力：二重請求。今日中の返金とメール連絡を希望。サービスは正常、解約はしない。',font=font(23),fill=INK)
  d.text((32,104),'AR：回答の穴を左から1つずつ',font=font(24,True),fill=GREEN)
  d.text((676,104),'拡散：全ての穴の予測を更新',font=font(24,True),fill=ORANGE)
  pa=panel(ar,k,'ar');pd=panel(df,k,'diffusion');im.paste(pa,(32,Y));im.paste(pd,(676,Y));ars.append(pa);dfs.append(pd)
  d.text((32,635),'黄色＝今回追加・変化した予測。確率は各穴の候補内で正規化した値。',font=font(21),fill=MUTED)
  d.text((32,667),'拡散の入力エンコードは表示外。各ステップの計算量・再生速度は実時間と異なります。',font=font(21),fill=MUTED)
  frames.append(im)
 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);dur=[1000]+[750]*n;dur[-1]=2600
 for name,ims in [('jev_comparison',frames),('ar_process',ars),('diffusion_jev_process',dfs)]:
  ims[0].save(out/(name+'.gif'),save_all=True,append_images=ims[1:],duration=dur,loop=0,optimize=True)
 frames[0].save(out/'jev_comparison_initial.png');frames[-1].save(out/'jev_comparison_final.png')
 print('Rendered',n+1,'frames; AR',len(ar['steps']),'forwards; diffusion',len(df['steps']),'decode forwards')
if __name__=='__main__':main()
