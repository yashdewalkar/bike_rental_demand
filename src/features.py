"""Shared training and inference feature preparation."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT = ['season', 'weathersit', 'holiday', 'workingday', 'weekday', 'month']
WEATHER = ['temp', 'atemp', 'hum', 'windspeed']
NUM = ['hour', 'year_offset', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'rush_hour'] + WEATHER
FEATURES = CAT + NUM


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw rows into forecast-time features only; never use rental counts."""
    df = raw.replace('?', np.nan).copy()
    date = pd.to_datetime(df['dteday'], format='%d-%m-%Y', errors='coerce')
    hour = pd.to_numeric(df['hr'], errors='coerce')
    if date.isna().any() or (~hour.between(0, 23)).any():
        raise ValueError('Every row must have a valid date and hour (0–23).')
    weekday = (date.dt.dayofweek + 1) % 7  # Sunday = 0 in supplied data.
    holiday = df['holiday'].where(df['holiday'].isin(['Yes', 'No']))
    working = df['workingday'].where(df['workingday'].isin(['Working Day', 'No work']))
    holiday = holiday.mask(holiday.isna() & working.eq('Working Day'), 'No')
    derived_work = pd.Series(np.where(weekday.isin([0, 6]) | holiday.eq('Yes'), 'No work', 'Working Day'), index=df.index)
    working = working.fillna(derived_work.where(holiday.notna()))
    season = df['season'].replace({'springer': 'spring'})
    # Missing season is safely inferred for unambiguous months only.
    season = season.fillna(date.dt.month.map({1: 'spring', 2: 'spring', 4: 'summer', 5: 'summer', 7: 'fall', 8: 'fall', 10: 'winter', 11: 'winter'}))
    x = pd.DataFrame({'season': season, 'weathersit': df['weathersit'], 'holiday': holiday, 'workingday': working,
                      'weekday': weekday.astype(str), 'month': date.dt.month.astype(str), 'hour': hour,
                      'year_offset': date.dt.year - 2011, 'hour_sin': np.sin(2 * np.pi * hour / 24),
                      'hour_cos': np.cos(2 * np.pi * hour / 24), 'month_sin': np.sin(2 * np.pi * date.dt.month / 12),
                      'month_cos': np.cos(2 * np.pi * date.dt.month / 12),
                      'rush_hour': (hour.between(7, 9) | hour.between(16, 19)).astype(int)}, index=df.index)
    for c in WEATHER:
        v = pd.to_numeric(df[c], errors='coerce')
        x[c] = v.where(v.between(0, 1))
    return x[FEATURES]


def load_data(path):
    df = pd.read_csv(path, na_values=['?']).drop_duplicates()
    df['timestamp'] = pd.to_datetime(df['dteday'], format='%d-%m-%Y', errors='coerce') + pd.to_timedelta(pd.to_numeric(df['hr'], errors='coerce'), unit='h')
    df['cnt'] = pd.to_numeric(df['cnt'], errors='coerce')
    df = df.loc[df['timestamp'].notna() & df['hr'].between(0, 23) & df['cnt'].ge(0)].copy()
    if df['timestamp'].duplicated().any():
        raise ValueError('Conflicting duplicate hourly observations found.')
    return df.sort_values('timestamp').reset_index(drop=True)
