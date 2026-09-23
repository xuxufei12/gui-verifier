"""Build transparent synthetic smoke data and a derived ScreenSpot evaluation only."""
import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = [{'id':'yes','text':'Yes'}, {'id':'no','text':'No'},
           {'id':'unknown','text':'Cannot determine from the screenshot'}]

def font(size):
    for name in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf']:
        if Path(name).exists(): return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)

def question(qid, text):
    return {'id':qid,'text':text,'options':OPTIONS,'unknown_option_id':'unknown'}

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')

def synthetic():
    out = ROOT / 'data/synthetic'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    states = ['success', 'login', 'success', 'error', 'success', 'loading']
    titles = {'train':['Workspace','Account portal','Team console'],
              'validation':['Member center'], 'test':['Project hub']}
    counts = {'train':96, 'validation':24, 'test':24}
    manifest = {}
    for split, count in counts.items():
        rng = random.Random(20260922 + list(counts).index(split))
        rows = []
        for i in range(count):
            state = states[i % len(states)]
            sid = f'{split}-{i:04d}'
            img = Image.new('RGB', (640,480), (237+rng.randrange(12),243,248))
            draw = ImageDraw.Draw(img)
            accent = rng.choice(['#173d64','#315b52','#49386b'])
            draw.rectangle((0,0,640,65), fill=accent)
            draw.text((24,16), rng.choice(titles[split]), font=font(24), fill='white')
            x, y = rng.randint(70,115), rng.randint(95,130)
            draw.rounded_rectangle((x,y,x+430,y+290),radius=12,fill='white',outline='#c8d3dd',width=2)
            if state == 'login':
                draw.text((x+24,y+20),'Sign in to continue',font=font(24),fill='#172b42')
                draw.text((x+24,y+70),'Email: member@example.test',font=font(18),fill='#4a5b6c')
                draw.text((x+24,y+115),'Password: ********',font=font(18),fill='#4a5b6c')
                button = 'Sign in'
                message = None
            elif state == 'success':
                draw.text((x+24,y+24),'Welcome back',font=font(25),fill='#172b42')
                message = 'Successfully signed in'
                button = 'Sign out'
            elif state == 'error':
                draw.text((x+24,y+24),'Sign in failed',font=font(25),fill='#a32626')
                message = 'Incorrect password'
                button = 'Try again'
            else:
                draw.text((x+24,y+24),'Signing in...',font=font(25),fill='#172b42')
                message = 'Please wait'
                button = 'Cancel'
            if message:
                draw.text((x+24,y+95),message,font=font(22),fill='#40576b')
            draw.rounded_rectangle((x+24,y+190,x+210,y+240),radius=7,fill=accent)
            draw.text((x+43,y+201),button,font=font(22),fill='white')
            # No label or state metadata is rendered outside the actual interface.
            img.save(out / 'images' / f'{sid}.png')
            # Balance yes/no across both success and non-success screenshots.
            buttons = ['Sign in','Sign out','Try again','Cancel']
            target = button if (i//2) % 2 == 0 else rng.choice([b for b in buttons if b != button])
            qs = [question('success','Does this screen explicitly confirm a successful login?'),
                  question('control',f'Is a button labeled "{target}" visible?'),
                  question('hidden','Has the backend database committed a session record for this login?')]
            rows.append({'id':sid,'group_id':sid,'source':'synthetic-rendered-ui-v1',
                         'image':f'images/{sid}.png',
                         'request':{'goal':f'Check login completion and locate the {target} button.',
                                    'rules':['Use visible screen evidence only. Backend persistence cannot be inferred.'],
                                    'questions':qs},
                         'labels':{'success':'yes' if state=='success' else 'no',
                                   'control':'yes' if button==target else 'no','hidden':'unknown'}})
        write_jsonl(out / f'{split}.jsonl', rows)
        manifest[split] = {'screenshots':len(rows), 'decisions':len(rows)*3,
                           'labels':dict(Counter(v for row in rows for v in row['labels'].values()))}
    (out/'manifest.json').write_text(json.dumps({'seed':20260922,'data_kind':'synthetic; not real GUI benchmark',
        'split_policy':'distinct rendered screens and application titles; task templates shared',
        'counts':manifest},indent=2),encoding='utf-8')
    print('Synthetic:',manifest,flush=True)

def screenspot(limit):
    raw = ROOT / 'data/ScreenSpot-v2'
    archive = raw / 'screenspotv2_image.zip'
    image_root = raw / 'images'
    image_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (image_root / info.filename).resolve()
            if not target.is_relative_to(image_root.resolve()): raise ValueError('Unsafe ZIP path')
        z.extractall(image_root)
    lookup = {p.name:p for p in image_root.rglob('*') if p.is_file()}
    candidates = []
    for file in sorted(raw.glob('screenspot_*_v2.json')):
        for row in json.loads(file.read_text(encoding='utf-8')):
            candidates.append((file.stem,row))
    random.Random(20260922).shuffle(candidates)
    out = ROOT / 'data/screenspot-verification'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for domain, row in candidates:
        name = row['img_filename']
        if name in seen: continue
        seen.add(name)
        path = lookup[name]
        source = Image.open(path).convert('RGB')
        x,y,w,h = row['bbox']
        if not (w>0 and h>0 and x>=0 and y>=0 and x+w<=source.width+1 and y+h<=source.height+1):
            raise ValueError(f'Unexpected bbox format: {row}')
        good = (int(x+w/2),int(y+h/2))
        locations = [(int(source.width*a),int(source.height*b)) for a,b in [(0.15,.15),(.85,.15),(.15,.85),(.85,.85),(.5,.5)]]
        bad = max(locations,key=lambda p:(p[0]-good[0])**2+(p[1]-good[1])**2)
        if x<=bad[0]<=x+w and y<=bad[1]<=y+h: continue
        group = hashlib.sha256(source.tobytes()).hexdigest()
        for suffix,point,label in [('positive',good,'yes'),('negative',bad,'no')]:
            img = source.copy()
            draw = ImageDraw.Draw(img)
            px,py=point
            radius=max(7,round(min(img.size)/80))
            draw.ellipse((px-radius,py-radius,px+radius,py+radius),outline='#ff00ff',width=3)
            draw.line((px-3,py,px+3,py),fill='#ff00ff',width=1)
            draw.line((px,py-3,px,py+3),fill='#ff00ff',width=1)
            sid=f'{len(seen):04d}-{suffix}'
            img.save(out/'images'/f'{sid}.png')
            rows.append({'id':sid,'group_id':group,'source':domain,'image':f'images/{sid}.png',
                         'request':{'goal':row['instruction'],'rules':['The magenta ring marks the proposed click point. Judge target matching, not task completion.'],
                                    'questions':[question('target','Does the center of the magenta ring point to the requested target?')]},
                         'labels':{'target':label},
                         'provenance':{'original_image':name,'bbox_xywh':row['bbox'],'point':point,
                                       'negative_policy':'far point outside annotated target; alternate valid targets not exhaustively annotated'}})
        if len(rows)>=limit*2: break
    write_jsonl(out/'test.jsonl',rows)
    (out/'manifest.json').write_text(json.dumps({'upstream':'OS-Copilot/ScreenSpot-v2',
        'license_tag':'Apache-2.0 (upstream)', 'usage':'evaluation only; derived candidate verification, NOT official ScreenSpot metric',
        'screenshots':len(rows),'source_images':len(rows)//2,'seed':20260922},indent=2),encoding='utf-8')
    print('ScreenSpot derived eval:',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--screenspot',action='store_true')
    p.add_argument('--limit',type=int,default=16)
    a=p.parse_args()
    synthetic()
    if a.screenspot: screenspot(a.limit)
