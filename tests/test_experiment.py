import pytest
from gui_verifier.evaluate import metrics
from gui_verifier.train import assert_disjoint


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
