"""Build a reviewed source-only release; never publish the working directory."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    files=[Path(x) for x in ['README.md','README.zh-CN.md','LICENSE','pyproject.toml','.gitignore','requirements-experiment004.txt',
        'docs/TRAINING.md','docs/RESULTS_004.md','docs/THIRD_PARTY.md','docs/RELEASE.md',
        'tools/train004.py','tools/evaluate004.py','tools/analyze004.py','tools/prepare_data.py','tools/build_public_release.py',
        'tests/test_contract.py','tests/test_data.py','tests/test_experiment.py','tests/test_experiment004.py']]
    for folder,pattern in [('src','*'),('examples','*.json'),('reports/published','*.json')]:
        files += [p.relative_to(ROOT) for p in (ROOT/folder).rglob(pattern) if p.is_file() and '__pycache__' not in p.parts]
    files=sorted(set(files))
    patterns=[r'172[.]30[.]',r'172[.]31[.]',r'feixu23',r'/root/autodl-tmp',r'invoke[.]feishudata',r'BEGIN [A-Z ]*PRIVATE KEY']
    # The scanner source contains its own test patterns, not deployment credentials.
    for rel in files:
        if rel.as_posix()=='tools/build_public_release.py':continue
        text=(ROOT/rel).read_text(encoding='utf-8')
        if any(re.search(p,text,re.I) for p in patterns):raise RuntimeError('Sensitive marker: '+str(rel))
    target=ROOT/'outputs/public-release'
    target.mkdir(parents=True,exist_ok=True)
    existing={p.relative_to(target) for p in target.rglob('*') if p.is_file() and '.git' not in p.relative_to(target).parts}
    extras=existing-set(files)-{Path('RELEASE_MANIFEST.json')}
    if extras:raise RuntimeError('Unexpected files in release directory: '+str(extras))
    manifest=[]
    for rel in files:
        dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
        manifest.append({'path':rel.as_posix(),'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
    (target/'RELEASE_MANIFEST.json').write_text(json.dumps({'scope':'source and aggregate reports only','files':manifest},indent=2),encoding='utf-8')
    archive=ROOT/'outputs/gui-verifier-public.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file() and '.git' not in p.relative_to(target).parts:z.write(p,p.relative_to(target).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'files':len(manifest),'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
