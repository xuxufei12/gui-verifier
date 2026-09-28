import os
import subprocess
import sys
from pathlib import Path

import pytest
from gui_verifier.evaluate import metrics
from gui_verifier.train import assert_disjoint

SRC = Path(__file__).resolve().parents[1] / 'src'


def test_metrics_have_clear_denominators():
    rows=[{'gold':'a','correct':True,'probabilities':{'a':.8,'b':.2},'max_probability':.8,'status':'decided'},
          {'gold':'b','correct':False,'probabilities':{'a':.6,'b':.4},'max_probability':.6,'status':'review'}]
    m=metrics(rows)
    assert m['accuracy']==.5
    assert m['coverage_decided']==.5
    assert m['error_rate_decided']==0
    assert m['multiclass_brier_sum']==pytest.approx(.4)
    assert m['ece_10_equal_width']==pytest.approx(.4)


def test_group_leakage_fails_closed():
    with pytest.raises(ValueError):
        assert_disjoint([{'group_id':'same'}],[{'group_id':'same'}])
    assert_disjoint([{'group_id':'one'}],[{'group_id':'two'}])


def test_training_entry_points_import_without_torch(tmp_path):
    """The suite must collect on a machine that has no CUDA stack installed."""
    for name in ('torch','peft'):
        (tmp_path/f'{name}.py').write_text(f"raise ImportError('{name} is not installed')\n",encoding='utf-8')
    program=('import gui_verifier.evaluate,gui_verifier.train\n'
             'from gui_verifier.train import assert_disjoint\n'
             "assert_disjoint([{'group_id':'a'}],[{'group_id':'b'}])\n"
             'print("ok")\n')
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(tmp_path),str(SRC)])}
    done=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,env=env)
    assert done.returncode==0,done.stderr
    assert done.stdout.strip()=='ok'
