from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import FEATURES, load_data, prepare_features

ROOT = Path(__file__).resolve().parents[1]


def sample():
    return pd.DataFrame([dict(dteday='01-01-2012', hr=17, season='springer', holiday='No', workingday='No work',
                              weathersit='Clear', temp=.5, atemp=.5, hum=.6, windspeed=.2)])


def test_no_target_leakage():
    raw = sample().assign(cnt=999, casual=100, registered=899, instant=7)
    x = prepare_features(raw)
    assert list(x.columns) == FEATURES
    assert not {'cnt', 'casual', 'registered', 'instant'}.intersection(x.columns)


def test_missing_values_and_calendar():
    x = prepare_features(sample().assign(temp='?', season='?', workingday='?'))
    assert x.loc[0, 'season'] == 'spring'
    assert x.loc[0, 'weekday'] == '0'
    assert x.loc[0, 'workingday'] == 'No work'
    assert pd.isna(x.loc[0, 'temp'])


def test_saved_model_inference():
    model = joblib.load(ROOT / 'artifacts/model.joblib')
    pred = model.predict(prepare_features(sample()))
    assert pred.shape == (1,) and np.isfinite(pred).all()


def test_data_integrity():
    df = load_data(ROOT / 'data/bike_rentals.csv')
    assert len(df) == 17379
    assert df.timestamp.is_monotonic_increasing
    assert df.timestamp.is_unique
