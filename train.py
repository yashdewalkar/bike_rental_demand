"""Reproduce chronological model selection, tuning, and held-out evaluation."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import sklearn
from src.features import CAT, NUM, load_data, prepare_features

ROOT = Path(__file__).resolve().parent
SEED = 42


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, np.maximum(0, pred))))


def make_pipeline(model):
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUM),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CAT),
    ])
    return Pipeline([('preprocess', preprocess), ('model', model)])


def main():
    df = load_data(ROOT / 'data/bike_rentals.csv')
    # Hold out the last 20% of calendar days, never split one day across partitions.
    days = df['timestamp'].dt.normalize().unique()
    cutoff = pd.Timestamp(days[int(len(days) * 0.8)])
    mask = df['timestamp'] < cutoff
    x, y = prepare_features(df), df['cnt']
    x_train, x_test, y_train, y_test = x[mask], x[~mask], y[mask], y[~mask]
    # Three expanding-window folds, each separated by chronological order.
    cv = TimeSeriesSplit(n_splits=3)
    candidates = {
        'Mean baseline': (DummyRegressor(), {}),
        'Ridge Regression': (Ridge(), {'model__alpha': [1, 10]}),
        'Decision Tree': (DecisionTreeRegressor(random_state=SEED), {'model__max_depth': [8, 14, 20], 'model__min_samples_leaf': [2, 8]}),
        'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1), {'model__max_depth': [12, 20], 'model__min_samples_leaf': [2, 5]}),
        'Gradient Boosting': (GradientBoostingRegressor(random_state=SEED), {'model__n_estimators': [150, 250], 'model__max_depth': [3, 5]}),
        'Histogram Gradient Boosting': (HistGradientBoostingRegressor(random_state=SEED, early_stopping=False), {'model__max_iter': [200, 400], 'model__max_leaf_nodes': [15, 31]}),
    }
    fits, records = {}, []
    scorer = make_scorer(rmse, greater_is_better=False)
    for name, (model, grid) in candidates.items():
        search = GridSearchCV(make_pipeline(model), grid, scoring=scorer, cv=cv, n_jobs=2, refit=True, error_score='raise')
        search.fit(x_train, y_train)
        fits[name] = search.best_estimator_
        records.append({'model': name, 'cv_rmse': -search.best_score_, 'best_params': json.dumps(search.best_params_, sort_keys=True)})
        print(f'{name}: CV RMSE {-search.best_score_:.2f}', flush=True)
    # Model choice is frozen before viewing any held-out errors.
    winner = min(records, key=lambda r: r['cv_rmse'])['model']
    for record in records:
        pred = np.maximum(0, fits[record['model']].predict(x_test))
        record.update(test_mae=float(mean_absolute_error(y_test, pred)), test_mse=float(mean_squared_error(y_test, pred)),
                      test_rmse=rmse(y_test, pred), test_r2=float(r2_score(y_test, pred)))
    results = pd.DataFrame(records).sort_values('cv_rmse')
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    (ROOT / 'reports').mkdir(exist_ok=True)
    results.to_csv(ROOT / 'reports/model_comparison.csv', index=False)
    model = fits[winner]
    joblib.dump(model, ROOT / 'artifacts/model.joblib', compress=3)
    pred = np.maximum(0, model.predict(x_test))
    pd.DataFrame({'timestamp': df.loc[~mask, 'timestamp'], 'actual': y_test, 'predicted': pred}).to_csv(ROOT / 'reports/test_predictions.csv', index=False)
    metadata = {'selected_model': winner, 'selection': 'lowest expanding-window CV RMSE', 'cutoff': str(cutoff.date()),
                'total_rows': len(df), 'train_rows': len(x_train), 'test_rows': len(x_test), 'sklearn_version': sklearn.__version__,
                'data_start': str(df.timestamp.min()), 'data_end': str(df.timestamp.max()), 'seed': SEED,
                'metrics': results.loc[results.model.eq(winner), ['cv_rmse', 'test_mae', 'test_mse', 'test_rmse', 'test_r2']].iloc[0].to_dict()}
    (ROOT / 'artifacts/metadata.json').write_text(json.dumps(metadata, indent=2))
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
