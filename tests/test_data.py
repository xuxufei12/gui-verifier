import importlib.util
from pathlib import Path
import json


def test_synthetic_group_split_and_label_balance(tmp_path):
    spec=importlib.util.spec_from_file_location('prepare_data',Path(__file__).resolve().parents[1]/'tools/prepare_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=tmp_path
    module.synthetic()
    groups=[]
    from gui_verifier.schema import VerifyRequest
    for split in ['train','validation','test']:
        rows=[json.loads(x) for x in (tmp_path/f'data/synthetic/{split}.jsonl').read_text().splitlines()]
        groups.append({r['group_id'] for r in rows})
        counts={'yes':0,'no':0,'unknown':0}
        for r in rows:
            VerifyRequest.model_validate(r['request'])
            assert (tmp_path/'data/synthetic'/r['image']).exists()
            for label in r['labels'].values():counts[label]+=1
        assert len(set(counts.values()))==1
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
