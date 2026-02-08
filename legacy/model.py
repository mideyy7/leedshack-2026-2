import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Regex to extract state/region from source_name, e.g. "Anand_VUNagar_DC (Gujarat)" -> "Gujarat"
LOC_RE = re.compile(r'\((.*)\)$')

FEAT = ["source_center", "dest_center", "hour", "e_time",
        "temp", "humidity", "pressure", "wind_gust"]


def extract_loc(name):
    if pd.isna(name):
        return None
    m = LOC_RE.search(str(name))
    return m.group(1) if m else None


def load_and_join(delivery_csv, weather_csv):
    """Load delivery and weather CSVs, join weather by location and nearest time (within 3h)."""
    delivery = pd.read_csv(delivery_csv)
    weather = pd.read_csv(weather_csv)

    delivery['od_start_time'] = pd.to_datetime(delivery['od_start_time'])
    weather['ts'] = pd.to_datetime(weather['ts'])

    delivery['loc'] = delivery['source_name'].apply(extract_loc)
    delivery = delivery.dropna(subset=['loc'])

    delivery = delivery.sort_values('od_start_time').reset_index(drop=True)
    weather = weather.sort_values('ts').reset_index(drop=True)

    merged = pd.merge_asof(
        delivery,
        weather,
        left_on='od_start_time',
        right_on='ts',
        by='loc',
        tolerance=pd.Timedelta('3h'),
        direction='nearest'
    )

    return merged


def build_df(merged, require_a_time=True):
    """Convert merged delivery+weather DataFrame to model-ready DataFrame."""
    df = pd.DataFrame({
        'shipment_id': merged['trip_uuid'],
        'ts': merged['od_start_time'],
        'source_center': merged['source_center'],
        'dest_center': merged['destination_center'],
        'hour': merged['od_start_time'].dt.hour,
        'e_time': pd.to_numeric(merged['osrm_time'], errors='coerce'),
        'temp': pd.to_numeric(merged['tempC'], errors='coerce'),
        'humidity': pd.to_numeric(merged['humidity'], errors='coerce'),
        'pressure': pd.to_numeric(merged['pressure'], errors='coerce'),
        'wind_gust': pd.to_numeric(merged['windGustKmph'], errors='coerce'),
        'a_time': pd.to_numeric(merged['actual_time'], errors='coerce'),
    })

    df['source_center'] = df['source_center'].astype('category')
    df['dest_center'] = df['dest_center'].astype('category')

    for col in ['temp', 'humidity', 'pressure', 'wind_gust', 'e_time']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        med = df[col].median() if df[col].notna().any() else 0.0
        df[col] = df[col].fillna(med)

    drop_subset = ['ts']
    if require_a_time:
        drop_subset.append('a_time')

    return df.dropna(subset=drop_subset).sort_values('ts').reset_index(drop=True)


def train_model(df, seed=42):
    """Train an LGBMRegressor on the given DataFrame and return the fitted model."""
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(df[FEAT], df["a_time"].astype(float).values)
    return model
