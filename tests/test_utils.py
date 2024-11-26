import pytest
from utils.preprocessing import log_transform, inverse_log_transform
import pandas as pd

def test_log_transform():
    df = pd.DataFrame({'value': [1, 2, 3]})
    df_log = log_transform(df, ['value'])
    
    assert df_log is not None
    assert 'value' in df_log.columns
    assert df_log['value'].iloc[0] == pytest.approx(0.693, 0.01)  # log1p(1) ~ 0.693

def test_inverse_log_transform():
    df_log = pd.DataFrame({'value': [0.693147, 1.098612, 1.386294]})  # log1p(1), log1p(2), log1p(3)
    df_orig = inverse_log_transform(df_log, ['value'])
    
    assert df_orig is not None
    assert 'value' in df_orig.columns
    assert df_orig['value'].iloc[0] == pytest.approx(1.0, 0.01)
