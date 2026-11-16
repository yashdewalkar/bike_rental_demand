"""Streamlit dashboard"""
from datetime import date
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.features import prepare_features

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Bike Demand Forecast', page_icon='🚲', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'artifacts/model.joblib')


def predict(raw):
    return np.maximum(0, load_model().predict(prepare_features(raw)))


st.title('🚲 Bike Rental Demand Forecast')
st.caption('Estimate hourly rental demand from calendar and weather conditions to support bike availability planning.')
if not (ROOT / 'artifacts/model.joblib').is_file():
    st.error('Model not found. Run python train.py before launching the app.')
    st.stop()

meta = json.loads((ROOT / 'artifacts/metadata.json').read_text())
metrics = meta['metrics']
cols = st.columns(4)
cols[0].metric('Selected model', meta['selected_model'])
cols[1].metric('Test RMSE', f"{metrics['test_rmse']:.1f}")
cols[2].metric('Test MAE', f"{metrics['test_mae']:.1f}")
cols[3].metric('Test R²', f"{metrics['test_r2']:.3f}")

forecast_tab, evaluation_tab, eda_tab = st.tabs(['Demand forecast', 'Model evaluation', 'Data insights'])
with forecast_tab:
    with st.form('forecast'):
        st.subheader('Forecast conditions')
        left, right = st.columns(2)
        with left:
            day = st.date_input('Date', value=date(2012, 12, 31), min_value=date(2011, 1, 1), max_value=date(2100, 12, 31))
            hour = st.slider('Hour of day', 0, 23, 17)
            season = st.selectbox('Season (dataset convention)', ['spring', 'summer', 'fall', 'winter'])
            holiday = st.checkbox('Public holiday')
            weather = st.selectbox('Weather situation', ['Clear', 'Mist', 'Light Snow', 'Heavy Rain'])
        with right:
            st.caption('Weather values are normalized to the dataset’s 0–1 scale.')
            temp = st.slider('Temperature (normalized)', 0.0, 1.0, .50, .01)
            atemp = st.slider('Feels-like temperature (normalized)', 0.0, 1.0, .50, .01)
            humidity = st.slider('Humidity (%)', 0, 100, 60)
            wind = st.slider('Wind speed (normalized)', 0.0, 1.0, .20, .01)
        submitted = st.form_submit_button('Predict demand', type='primary')
    if submitted:
        raw = {'dteday': day.strftime('%d-%m-%Y'), 'hr': hour, 'season': season, 'holiday': 'Yes' if holiday else 'No',
               'workingday': 'No work' if holiday or day.weekday() >= 5 else 'Working Day', 'weathersit': weather,
               'temp': temp, 'atemp': atemp, 'hum': humidity / 100, 'windspeed': wind}
        value = float(predict(pd.DataFrame([raw]))[0])
        daily = pd.DataFrame([{**raw, 'hr': h} for h in range(24)])
        daily['Predicted rentals'] = predict(daily)
        st.metric('Estimated rentals at selected hour', f'{value:,.0f} bikes')
        st.subheader('Hourly demand profile')
        st.line_chart(daily.set_index('hr')['Predicted rentals'], x_label='Hour', y_label='Estimated rentals')
        st.caption('Daily profile assumes weather is constant throughout the day. Predictions are expected rentals, not inventory requirements.')
        st.download_button('Download prediction profile', daily[['hr', 'Predicted rentals']].to_csv(index=False), 'hourly_forecast.csv', 'text/csv')
        if day.year not in [2011, 2012]:
            st.warning('This date is beyond the historical training era (2011–2012). Treat the forecast as an illustrative scenario, not a validated future prediction.')
        if weather == 'Heavy Rain':
            st.warning('Heavy rain is underrepresented in the historical dataset, so the estimate is less reliable.')
with evaluation_tab:
    st.subheader('Chronological holdout evaluation')
    st.write(f"Training: {meta['train_rows']:,} rows | Test: {meta['test_rows']:,} rows | Test begins: {meta['cutoff']}")
    results = pd.read_csv(ROOT / 'reports/model_comparison.csv')
    st.dataframe(results.drop(columns='best_params').round(3), use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index('model')['cv_rmse'], y_label='CV RMSE (lower is better)')
    p = pd.read_csv(ROOT / 'reports/test_predictions.csv', parse_dates=['timestamp']).set_index('timestamp')
    st.subheader('Actual versus predicted daily rentals')
    st.line_chart(p.resample('D').sum())
    st.caption('The model was selected using expanding-window CV on training data; the test period was reserved for reporting.')
with eda_tab:
    st.subheader('Dataset overview')
    summary = json.loads((ROOT / 'reports/eda_summary.json').read_text())
    st.write(f"{summary['rows']:,} hourly observations · {summary['columns']} original columns · {summary['duplicate_rows']} exact duplicates")
    for image in ['eda_overview.png', 'correlation.png', 'temperature_scatter.png']:
        st.image(str(ROOT / 'reports' / image))
    st.write('Missing values (including question-mark placeholders)')
    st.dataframe(pd.Series(summary['missing_values'], name='Missing values').to_frame(), use_container_width=True)
