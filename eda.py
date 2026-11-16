"""Create a reproducible EDA report and charts from the supplied dataset."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.features import load_data, prepare_features

ROOT = Path(__file__).resolve().parent


def main():
    raw = pd.read_csv(ROOT / 'data/bike_rentals.csv', na_values=['?'])
    df = load_data(ROOT / 'data/bike_rentals.csv')
    x = prepare_features(df)
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    q1, q3 = df.cnt.quantile([.25, .75]); iqr = q3 - q1
    summary = dict(rows=len(raw), columns=len(raw.columns), duplicate_rows=int(raw.duplicated().sum()),
                   usable_rows=len(df), missing_values=raw.isna().sum().to_dict(), target_mean=float(df.cnt.mean()),
                   target_median=float(df.cnt.median()), target_max=int(df.cnt.max()),
                   target_iqr_outliers=int(((df.cnt < q1 - 1.5 * iqr) | (df.cnt > q3 + 1.5 * iqr)).sum()),
                   outlier_policy='Retain valid high-demand observations; median imputation for missing numeric features in training folds.')
    (out / 'eda_summary.json').write_text(json.dumps(summary, indent=2))
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    ax[0, 0].hist(df.cnt, bins=40, color='#287271'); ax[0, 0].set(title='Hourly rental count distribution', xlabel='Rentals', ylabel='Hours')
    df.groupby('hr').cnt.mean().plot(ax=ax[0, 1], marker='o', color='#287271'); ax[0, 1].set(title='Average demand by hour', xlabel='Hour', ylabel='Rentals')
    df.set_index('timestamp').cnt.resample('D').sum().rolling(7).mean().plot(ax=ax[1, 0], color='#e76f51'); ax[1, 0].set(title='Daily demand (7-day rolling mean)', ylabel='Rentals/day')
    df.groupby('season').cnt.mean().plot.bar(ax=ax[1, 1], color='#287271'); ax[1, 1].set(title='Average demand by season', ylabel='Rentals/hour'); ax[1, 1].tick_params(axis='x', rotation=0)
    fig.savefig(out / 'eda_overview.png', dpi=150); plt.close(fig)
    numeric = x[['temp', 'atemp', 'hum', 'windspeed', 'hour', 'year_offset']].assign(cnt=df.cnt)
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained'); im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right'); ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax); ax.set_title('Pearson correlation (forecast-time features only)'); fig.savefig(out / 'correlation.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained'); ax.scatter(x.temp, df.cnt, s=5, alpha=.15, color='#287271'); ax.set(title='Temperature versus rental demand', xlabel='Normalized temperature', ylabel='Rentals/hour'); fig.savefig(out / 'temperature_scatter.png', dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
