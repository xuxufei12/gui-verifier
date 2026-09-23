"""Render dataset input images from locally acquired upstream originals."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def render(source,row,font_regular,font_bold):
    with Image.open(source) as im:image=im.convert('RGB')
    draw=ImageDraw.Draw(image)
    for a in row['markers']:
        box=[round(v*(image.width if i%2==0 else image.height)) for i,v in enumerate(a['bbox_xyxy_normalized'])]
        x1,y1,x2,y2=box;label=a['marker']
        draw.rectangle(box,outline='#ff00ff',width=max(2,round(image.width/700)))
        if row['marker_version']=='inside_top_left_v2':
            size=max(8,min(round(min(image.size)/65),max(8,y2-y1-4),max(8,x2-x1-4)))
            font=ImageFont.truetype(str(font_bold),size)
            b=draw.textbbox((0,0),label,font=font);bw,bh=b[2]-b[0]+4,b[3]-b[1]+4
            draw.rectangle((x1,y1,x1+bw,y1+bh),fill='#ff00ff',outline='black',width=1)
            draw.text((x1+2-b[0],y1+2-b[1]),label,font=font,fill='black')
        else:
            font=ImageFont.truetype(str(font_regular),max(14,round(min(image.size)/55)))
            y=max(0,y1-font.size-3)
            draw.rectangle((x1,y,x1+font.size+4,y+font.size+3),fill='#ff00ff')
            draw.text((x1+2,y),label,font=font,fill='black')
    return image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='datasets/gui-verifier-v1');p.add_argument('--source-dir',required=True)
    p.add_argument('--font-regular',required=True);p.add_argument('--font-bold',required=True)
    a=p.parse_args();root=Path(a.dataset);index={}
    for f in Path(a.source_dir).rglob('*'):
        if f.is_file() and f.suffix.lower() in ['.png','.jpg','.jpeg']:index.setdefault(f.stem,[]).append(f)
    rows=[json.loads(l) for l in (root/'images-manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    report={'matched':0,'missing':[],'different_render_hash':[]}
    for row in rows:
        dst=root/row['image'];candidates=index.get(row['id'],[])+index.get(Path(row['original_name']).stem,[])
        source=next((f for f in candidates if hashlib.sha256(f.read_bytes()).hexdigest()==row['original_sha256']),None)
        if source is None:report['missing'].append(row['id']);continue
        if dst.exists():
            if hashlib.sha256(dst.read_bytes()).hexdigest()==row['image_sha256']:report['matched']+=1;continue
            raise RuntimeError('Refusing to overwrite '+str(dst))
        dst.parent.mkdir(parents=True,exist_ok=True)
        render(source,row,a.font_regular,a.font_bold).save(dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest()==row['image_sha256']:report['matched']+=1
        else:report['different_render_hash'].append(row['id'])
    (root/'restore-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));raise SystemExit(1 if report['missing'] or report['different_render_hash'] else 0)
if __name__=='__main__':main()
