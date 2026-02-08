from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
import numpy as np
import lightgbm as lgb
import joblib
import re
import os
import time
import json
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair as SolanaKeypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.transaction import Transaction as SolanaTransaction
from solders.message import Message as SolanaMessage

app = FastAPI(title="Chain-Reaction Supply Chain API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data
graph = None
shipments_df = None
weather_df = None
weather_loc_avg = None
weather_global_avg = None
risk_classifier = None
delay_regressor = None
model_meta = None  # stores split info + feature metadata
predictions_cache = None  # dict of trip_uuid -> {risk, expected_delay_hours}
simulation_cache = None   # dict of trip_uuid -> {best_case, worst_case, expected_delay_hours, p10, p90}
mitigation_cache = None   # dict of trip_uuid -> {strategy, expected_risk_reduction, solana_tx}
story_cache = None        # dict of trip_uuid -> {story, previous_state, current_state, risk, ...}
analysis_executed = False  # True after /run-analysis has been called

# Solana config
SOLANA_RPC_URL = "https://api.devnet.solana.com"
MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
HIGH_RISK_THRESHOLD = 0.2
SOLANA_ENABLED = os.environ.get("SOLANA_ENABLED", "true").lower() == "true"

# Global Wallet (Initialize as None, setup on startup)
authority_keypair = None 
solana_client = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_shipments():
    """Load delivery CSV and build a directed supply chain graph."""
    global graph, shipments_df

    csv_path = os.path.join(DATA_DIR, "delivery.csv")
    if not os.path.exists(csv_path):
        print("WARNING: delivery.csv not found at", csv_path)
        return

    shipments_df = pd.read_csv(csv_path)
    print(f"Loaded {len(shipments_df)} shipment records")

    # Build directed graph from unique source -> destination pairs
    graph = nx.DiGraph()

    # Get unique edges with aggregated attributes
    edges = shipments_df.groupby(["source_name", "destination_name"]).agg(
        transit_time=("osrm_time", "median"),
        actual_time_fallback=("actual_time", "median"),
        distance=("actual_distance_to_destination", "median"),
    ).reset_index()

    # Use osrm_time as transit_time; fallback to actual_time if missing
    edges["transit_time"] = edges["transit_time"].fillna(edges["actual_time_fallback"])
    edges.drop(columns=["actual_time_fallback"], inplace=True)

    # Add edges with attributes
    for _, row in edges.iterrows():
        graph.add_edge(
            row["source_name"],
            row["destination_name"],
            transit_time=round(float(row["transit_time"]), 2),
            distance=round(float(row["distance"]), 2),
        )

    # Set default node attributes
    for node in graph.nodes:
        graph.nodes[node]["state"] = "on-time"
        graph.nodes[node]["historical_delay_prob"] = 0.0

    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


def _extract_city(node_name):
    """Extract base city name from node like 'Agra_Central_D_3 (Uttar Pradesh)' -> 'agra'."""
    name = node_name.split("(")[0].strip()
    return name.split("_")[0].lower()


def _extract_state(node_name):
    if pd.isna(node_name):
        return None
    
    """Extract state from node like 'Agra_Central_D_3 (Uttar Pradesh)' -> 'Uttar Pradesh'."""
    match = re.search(r"\(([^)]+)\)", node_name)
    return match.group(1) if match else None


WEATHER_ATTRS = ["temperature", "humidity", "wind_speed", "rain_1h", "snow_1h", "weather_main"]


def load_weather():
    """Load weather2 CSV and merge weather attributes onto graph nodes (state-level averages)."""
    global weather_df, weather_loc_avg, weather_global_avg

    csv_path = os.path.join(DATA_DIR, "weather2.csv")
    if not os.path.exists(csv_path):
        print("WARNING: weather2.csv not found at", csv_path)
        return

    raw = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(raw)} weather records")

    raw["ts"] = pd.to_datetime(raw["ts"], errors="coerce")
    raw["loc"] = raw["loc"].astype(str).str.strip()
    raw = raw.dropna(subset=["loc", "ts"])

    # Normalize schema to the expected weather feature names
    weather_df = raw.groupby(["loc", "ts"]).agg(
        temperature=("tempC", "mean"),
        humidity=("humidity", "mean"),
        wind_speed=("windGustKmph", "mean"),
        rain_1h=("precipMM", "mean"),
        weather_main=("weatherDescValue", lambda s: s.dropna().iloc[0] if len(s.dropna()) else "Clear"),
    ).reset_index()
    weather_df["snow_1h"] = 0.0

    # Precompute location and global averages for fallback during merges
    weather_loc_avg = weather_df.groupby("loc").agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"),
        rain_1h=("rain_1h", "mean"),
        weather_main=("weather_main", lambda s: s.mode().iloc[0] if len(s.mode()) else "Clear"),
    ).reset_index()
    weather_loc_avg["snow_1h"] = 0.0

    weather_global_avg = {
        "temperature": float(weather_df["temperature"].mean()),
        "humidity": float(weather_df["humidity"].mean()),
        "wind_speed": float(weather_df["wind_speed"].mean()),
        "rain_1h": float(weather_df["rain_1h"].mean()),
        "snow_1h": 0.0,
        "weather_main": "Clear",
    }

    if graph is None:
        print("WARNING: Graph not built yet, skipping weather merge")
        return

    # Merge state-level averages onto graph nodes
    node_states = {node: _extract_state(node) for node in graph.nodes}
    matched_state, matched_fallback = 0, 0

    loc_avg_lookup = {
        row["loc"]: row for _, row in weather_loc_avg.iterrows()
    }

    for node in graph.nodes:
        state = node_states.get(node)
        if state and state in loc_avg_lookup:
            row = loc_avg_lookup[state]
            for attr in WEATHER_ATTRS:
                val = row[attr]
                graph.nodes[node][attr] = round(float(val), 2) if attr != "weather_main" else str(val)
            matched_state += 1
            continue

        for attr in WEATHER_ATTRS:
            graph.nodes[node][attr] = weather_global_avg[attr] if attr != "weather_main" else weather_global_avg["weather_main"]
        matched_fallback += 1

    print(f"Weather merged: {matched_state} state-level, {matched_fallback} fallback")


# --- Phase 4: Feature engineering & LightGBM training ---

# NOTE: Exclude leakage features derived from actual outcomes (e.g., factor/segment_factor).
FEATURE_COLS = [
    "osrm_time", "actual_distance_to_destination", "segment_osrm_time",
    "segment_osrm_distance",
    "temperature", "humidity", "wind_speed", "rain_1h", "snow_1h",
    "route_type_encoded", "weather_main_encoded",
    "hour_of_day", "day_of_week", "is_peak_hour",
]

def _nearest_3h_bucket_minutes(minute_of_day):
    """Round to nearest 3-hour bucket (00, 03, ..., 21) without crossing day boundaries."""
    bucket = ((minute_of_day + 89) // 180) * 180
    return bucket.clip(lower=0, upper=21 * 60)

def _attach_weather_for_shipments(df):
    """Attach weather2 features to shipment rows using closest 3-hour bucket."""
    if weather_df is None or weather_df.empty:
        for col in ["temperature", "humidity", "wind_speed", "rain_1h", "snow_1h"]:
            df[col] = 0.0
        df["weather_main"] = "Clear"
        return df

    df["source_state"] = df["source_name"].map(_extract_state)

    minute_of_day = (df["od_start_time_parsed"].dt.hour * 60 + df["od_start_time_parsed"].dt.minute)
    minute_of_day = minute_of_day.fillna(12 * 60).astype(int)
    bucket_minutes = _nearest_3h_bucket_minutes(minute_of_day)
    bucket_hours = (bucket_minutes // 60).astype(int)

    default_date = weather_df["ts"].min().normalize()
    date_base = df["od_start_time_parsed"].dt.normalize().fillna(default_date)
    df["weather_ts"] = date_base + pd.to_timedelta(bucket_hours, unit="h")

    weather_cols = ["loc", "ts", "temperature", "humidity", "wind_speed", "rain_1h", "snow_1h", "weather_main"]
    df = df.merge(
        weather_df[weather_cols],
        how="left",
        left_on=["source_state", "weather_ts"],
        right_on=["loc", "ts"],
    ).drop(columns=["loc", "ts"], errors="ignore")

    # Fallback to state averages when exact time match is missing
    if weather_loc_avg is not None and not weather_loc_avg.empty:
        loc_avg = weather_loc_avg.set_index("loc")
        for col in ["temperature", "humidity", "wind_speed", "rain_1h", "snow_1h"]:
            df[col] = df[col].fillna(df["source_state"].map(loc_avg[col]))
        df["weather_main"] = df["weather_main"].fillna(df["source_state"].map(loc_avg["weather_main"]))

    # Global fallback
    if weather_global_avg:
        for col in ["temperature", "humidity", "wind_speed", "rain_1h", "snow_1h"]:
            df[col] = df[col].fillna(weather_global_avg[col])
        df["weather_main"] = df["weather_main"].fillna(weather_global_avg["weather_main"])

    return df

def _build_feature_df():
    """Build a feature DataFrame by merging shipment rows with weather2.csv attributes."""
    df = shipments_df.copy()

    # Parse temporal features from od_start_time
    df["od_start_time_parsed"] = pd.to_datetime(df["od_start_time"], format="mixed", dayfirst=False, errors="coerce")
    df["hour_of_day"] = df["od_start_time_parsed"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["od_start_time_parsed"].dt.dayofweek.fillna(0).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].apply(lambda h: 1 if 7 <= h <= 10 or 16 <= h <= 20 else 0)

    # Encode categoricals
    route_map = {rt: i for i, rt in enumerate(df["route_type"].dropna().unique())}
    df["route_type_encoded"] = df["route_type"].map(route_map).fillna(0).astype(int)

    # Merge weather from weather2.csv using closest 3-hour bucket to od_start_time
    df = _attach_weather_for_shipments(df)

    weather_main_categories = sorted(df["weather_main"].dropna().unique())
    weather_main_map = {wm: i for i, wm in enumerate(weather_main_categories)}

    # --- CHANGE 1: Weather Dampening ---
    # Reduce impact of rain/snow so it doesn't immediately spike risk to 100%
    df["rain_1h"] = df["rain_1h"] * 0.4
    df["snow_1h"] = df["snow_1h"] * 0.4 
    
    df["weather_main_raw"] = df["weather_main"].fillna("Clear")
    df["weather_main_encoded"] = df["weather_main_raw"].map(weather_main_map).fillna(0).astype(int)

    # --- CHANGE 2: Stricter Thresholds ---
    # Only flag as delayed if it took > 2.0x the expected time (was 1.5)
    df["is_delayed"] = (df["factor"] > 2.0).astype(int)

    # Regression: delay in hours
    df["delay_hours"] = ((df["actual_time"] - df["osrm_time"]) / 60).clip(lower=0)

    # Fill NaN in feature columns
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df, weather_main_map

def _time_based_split(df, test_fraction=0.3):
    """Return (train_df, test_df, cutoff_ts)."""
    df = df.sort_values("od_start_time_parsed").dropna(subset=["od_start_time_parsed"])
    if len(df) == 0:
        return df, df, None
    split_idx = int(len(df) * (1 - test_fraction))
    cutoff_ts = df.iloc[split_idx]["od_start_time_parsed"]
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test, cutoff_ts


def train_models():
    """Train LightGBM classifier and regressor, save to disk."""
    global risk_classifier, delay_regressor, model_meta

    os.makedirs(MODEL_DIR, exist_ok=True)
    clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
    reg_path = os.path.join(MODEL_DIR, "regressor.pkl")
    meta_path = os.path.join(MODEL_DIR, "meta.pkl")

    # If models already exist, load them
    if os.path.exists(clf_path) and os.path.exists(reg_path):
        risk_classifier = joblib.load(clf_path)
        delay_regressor = joblib.load(reg_path)
        if os.path.exists(meta_path):
            model_meta = joblib.load(meta_path)
        print("Loaded existing models from disk")
        return

    print("Training LightGBM models...")
    df, weather_main_map = _build_feature_df()

    # Time-based split to avoid leakage across time
    train_df, test_df, cutoff_ts = _time_based_split(df, test_fraction=0.3)
    X_train = train_df[FEATURE_COLS]
    y_cls = train_df["is_delayed"].values
    y_reg = train_df["delay_hours"].values

    # Train classifier
    risk_classifier = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        num_leaves=31, verbose=-1, n_jobs=-1,
    )
    risk_classifier.fit(X_train, y_cls)

    # Train regressor
    delay_regressor = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        num_leaves=31, verbose=-1, n_jobs=-1,
    )
    delay_regressor.fit(X_train, y_reg)

    # Save models and metadata
    joblib.dump(risk_classifier, clf_path)
    joblib.dump(delay_regressor, reg_path)
    model_meta = {
        "weather_main_map": weather_main_map,
        "split_cutoff_ts": cutoff_ts,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "feature_cols": FEATURE_COLS,
    }
    joblib.dump(model_meta, meta_path)

    print(
        f"Models trained and saved. "
        f"Delay rate: {y_cls.mean():.2%}, Mean delay: {y_reg.mean():.1f}h, "
        f"Train/Test: {len(train_df)}/{len(test_df)}"
    )

def compute_predictions():
    """Run predictions for all shipments and cache results."""
    global predictions_cache

    if risk_classifier is None or delay_regressor is None:
        print("WARNING: Models not loaded, skipping predictions")
        return

    df, _ = _build_feature_df()
    X = df[FEATURE_COLS]

    # 1. Get Raw Probabilities & Delays
    raw_risk_probs = risk_classifier.predict_proba(X)[:, 1]
    raw_delay_hours = delay_regressor.predict(X)

    df["raw_risk"] = raw_risk_probs
    df["predicted_delay_hours"] = np.clip(raw_delay_hours, 0, None)

    # --- NEW LOGIC: SEVERITY ADJUSTMENT ---
    
    # Define a "Critical Delay Threshold" (e.g., 2 hours). 
    # If delay < 2 hours, we dampen the risk score.
    # If delay >= 2 hours, we keep the full risk score.
    CRITICAL_DELAY_HOURS = 2.0
    
    # Create a multiplier: 0.1h delay -> 0.05 multiplier; 2.0h delay -> 1.0 multiplier
    # We add a small base (0.2) so even small delays satisfy "some" risk, but not 95%
    df["severity_multiplier"] = (df["predicted_delay_hours"] / CRITICAL_DELAY_HOURS).clip(0.2, 1.0)
    
    # Calculate Final Risk: Certainty * Severity
    df["adjusted_risk"] = df["raw_risk"] * df["severity_multiplier"]

    # --------------------------------------

    # Aggregate per trip (Weighted Average Logic from previous step)
    trip_totals = df.groupby("trip_uuid")["segment_osrm_distance"].transform("sum")
    df["segment_weight"] = df["segment_osrm_distance"] / trip_totals.replace(0, 1)
    
    df["weighted_risk"] = df["adjusted_risk"] * df["segment_weight"]

    trip_preds = df.groupby("trip_uuid").agg(
        risk=("weighted_risk", "sum"),
        expected_delay_hours=("predicted_delay_hours", "sum"), # Summing delay across segments
    )

    # Add stochastic noise (Jitter) for simulation realism
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.03, size=len(trip_preds)) # Small 3% jitter
    trip_preds["risk"] = (trip_preds["risk"] + noise).clip(0.01, 0.99)

    predictions_cache = {
        uid: {
            "risk": round(float(row["risk"]), 4),
            "expected_delay_hours": round(float(row["expected_delay_hours"]), 2),
        }
        for uid, row in trip_preds.iterrows()
    }

    print(f"Predictions computed for {len(predictions_cache)} unique trips")

# --- Phase 6: Monte Carlo simulation ---

N_SIMULATIONS = 100

def run_simulations():
    """Run Monte Carlo simulations for all trips using the trained regressor."""
    global simulation_cache

    if delay_regressor is None:
        print("WARNING: Regressor not loaded, skipping simulations")
        return

    df, _ = _build_feature_df()
    rng = np.random.default_rng(42)

    perturb_cols = ["osrm_time", "segment_osrm_time", "temperature", "humidity",
                    "wind_speed", "rain_1h", "snow_1h"]

    X_base = df[FEATURE_COLS].values.copy()

    weather_severity = (
        df["rain_1h"].values * 0.3
        + df["snow_1h"].values * 0.5
        + df["wind_speed"].values * 0.02
    ).clip(0, 1)

    all_delays = np.zeros((len(df), N_SIMULATIONS))
    col_indices = [FEATURE_COLS.index(c) for c in perturb_cols]

    for sim in range(N_SIMULATIONS):
        X_sim = X_base.copy()
        for ci in col_indices:
            # --- CHANGE 4: Increased Variance ---
            # Increase noise scale from 0.2 to 0.3 for more "drama" in the simulation
            noise_scale = 0.3 * (1 + weather_severity)
            noise = rng.normal(1.0, noise_scale, size=len(df))
            X_sim[:, ci] = X_base[:, ci] * noise

        X_sim_df = pd.DataFrame(X_sim, columns=FEATURE_COLS)
        preds = delay_regressor.predict(X_sim_df)
        all_delays[:, sim] = np.clip(preds, 0, None)

    # Aggregate per trip_uuid
    df["best_case"] = np.min(all_delays, axis=1)
    df["worst_case"] = np.max(all_delays, axis=1)
    df["sim_expected"] = np.mean(all_delays, axis=1)
    df["p10"] = np.percentile(all_delays, 10, axis=1)
    df["p90"] = np.percentile(all_delays, 90, axis=1)

    trip_sims = df.groupby("trip_uuid").agg(
        best_case=("best_case", "min"),
        worst_case=("worst_case", "max"),
        expected_delay_hours=("sim_expected", "mean"),
        p10=("p10", "min"),
        p90=("p90", "max"),
    )

    simulation_cache = {
        uid: {
            "best_case": round(float(row["best_case"]), 2),
            "worst_case": round(float(row["worst_case"]), 2),
            "expected_delay_hours": round(float(row["expected_delay_hours"]), 2),
            "p10": round(float(row["p10"]), 2),
            "p90": round(float(row["p90"]), 2),
        }
        for uid, row in trip_sims.iterrows()
    }

    print(f"Monte Carlo simulations done for {len(simulation_cache)} trips ({N_SIMULATIONS} scenarios each)")

# --- Phase 7: Mitigation with Solana ---

MITIGATION_STRATEGIES = [
    {"strategy": "Reroute shipment via alternate corridor", "expected_risk_reduction": 0.25},
    {"strategy": "Expedite critical leg with express carrier", "expected_risk_reduction": 0.20},
    {"strategy": "Hold inventory upstream at distribution center", "expected_risk_reduction": 0.15},
]


def _select_strategy(trip_uuid, risk):
    """Pick a mitigation strategy with hash-based diversity across all 3 strategies."""
    h = hash(trip_uuid) % 3
    if risk > 0.8:
        idx = h
    elif risk > 0.5:
        idx = (h + 1) % 3
    else:
        idx = (h + 2) % 3
    return MITIGATION_STRATEGIES[idx]


def _trigger_solana_memo(trip_uuid, risk, action_desc):
    """
    Send a memo transaction to Solana devnet.
    Uses the global authority_keypair to sign.
    """
    global solana_client, authority_keypair

    if not SOLANA_ENABLED or solana_client is None or authority_keypair is None:
        return None

    try:
        # Create the log message
        # Format: AppName | ID | Risk | Action
        memo_text = f"ChainReaction|{trip_uuid}|risk:{risk:.2f}|{action_desc}"
        
        # Ensure message fits within Memo program limits
        memo_bytes = memo_text.encode("utf-8")
        
        # 1. Create the Instruction
        ix = Instruction(
            program_id=MEMO_PROGRAM_ID,
            accounts=[
                AccountMeta(pubkey=authority_keypair.pubkey(), is_signer=True, is_writable=True)
            ],
            data=memo_bytes,
        )

        # 2. Get Recent Blockhash
        latest_blockhash_resp = solana_client.get_latest_blockhash()
        recent_blockhash = latest_blockhash_resp.value.blockhash

        # 3. Create Transaction Message
        msg = SolanaMessage.new_with_blockhash(
            [ix], 
            authority_keypair.pubkey(), 
            recent_blockhash
        )

        # 4. Sign and Build Transaction
        tx = SolanaTransaction.new_unsigned(msg)
        tx.sign([authority_keypair], recent_blockhash)

        # 5. Send Transaction
        # skip_preflight=True is faster but risks failure if logic is wrong. 
        # For memos, it's usually safe.
        result = solana_client.send_transaction(tx)
        
        sig = str(result.value)
        print(f"Solana Log Success: https://explorer.solana.com/tx/{sig}?cluster=devnet")
        return sig

    except Exception as e:
        print(f"Solana tx failed for {trip_uuid}: {e}")
        return None

def compute_mitigations():
    """Generate mitigation strategies for high-risk shipments, optionally trigger Solana."""
    global mitigation_cache

    if predictions_cache is None:
        print("WARNING: Predictions not available, skipping mitigations")
        return

    mitigation_cache = {}
    high_risk = {uid: p for uid, p in predictions_cache.items() if p["risk"] > HIGH_RISK_THRESHOLD}

    print(f"Computing mitigations for {len(high_risk)} high-risk shipments...")

    # Only send Solana txs for top 10 riskiest (demo budget — devnet airdrop is rate-limited)
    sorted_high = sorted(high_risk.items(), key=lambda x: x[1]["risk"], reverse=True)
    top_for_chain = set(uid for uid, _ in sorted_high[:10]) if SOLANA_ENABLED else set()

    rng = np.random.default_rng(42)

    for uid, pred in high_risk.items():
        strat = _select_strategy(uid, pred["risk"])

        # Add controlled variability to risk reduction (±5%)
        base_reduction = strat["expected_risk_reduction"]
        variation = rng.uniform(-0.05, 0.05)
        actual_reduction = round(max(0.05, base_reduction + variation), 4)

        solana_tx = None
        if uid in top_for_chain:
            solana_tx = _trigger_solana_memo(uid, pred["risk"], strat["strategy"])
            time.sleep(1)

        mitigation_cache[uid] = {
            "strategy": strat["strategy"],
            "original_risk": round(pred["risk"], 4),
            "expected_risk_reduction": actual_reduction,
            "mitigated_risk": round(max(0, pred["risk"] - actual_reduction), 4),
            "solana_tx": solana_tx,
        }

    on_chain_count = sum(1 for m in mitigation_cache.values() if m["solana_tx"])
    print(f"Mitigations computed: {len(mitigation_cache)} strategies, {on_chain_count} on-chain txs")


# --- Phase 8: Journey Stories ---

def _determine_state(risk):
    """Classify shipment state based on risk score."""
    if risk > 0.7:
        return "delayed"
    elif risk > 0.4:
        return "at-risk"
    return "on-time"


def _get_weather_cause(trip_uuid):
    """Find the dominant weather factor for a shipment's source at the closest 3-hour bucket."""
    if shipments_df is None:
        return None

    rows = shipments_df[shipments_df["trip_uuid"] == trip_uuid]
    if rows.empty:
        return None

    row = rows.iloc[0].copy()
    df = pd.DataFrame([row])
    df["od_start_time_parsed"] = pd.to_datetime(df["od_start_time"], format="mixed", dayfirst=False, errors="coerce")
    df = _attach_weather_for_shipments(df)
    attrs = df.iloc[0].to_dict()

    rain = float(attrs.get("rain_1h", 0))
    snow = float(attrs.get("snow_1h", 0))
    wind = float(attrs.get("wind_speed", 0))
    weather = attrs.get("weather_main", "Clear")

    if snow > 0:
        return f"snowfall ({weather})"
    if rain > 0.5:
        return f"heavy rain ({weather})"
    if wind > 10:
        return f"high winds ({weather})"
    if rain > 0:
        return f"rain ({weather})"
    if weather not in ("Clear", "Clouds"):
        return weather.lower()
    return None

def setup_solana_wallet():
    """Initialize the global wallet and fund it if necessary."""
    global authority_keypair, solana_client
    
    if not SOLANA_ENABLED:
        print("Solana integration disabled via config.")
        return

    try:
        solana_client = SolanaClient(SOLANA_RPC_URL)
        
        # --- FIXED LOADING LOGIC ---
        if os.path.exists("authority_keypair.json"):
            with open("authority_keypair.json", "r") as f:
                raw = f.read()
                # Load the list of integers from JSON
                key_list = json.loads(raw)
                # Convert list back to bytes for the Keypair constructor
                authority_keypair = SolanaKeypair.from_bytes(bytes(key_list))
                print(f"Loaded existing Solana Wallet: {authority_keypair.pubkey()}")
        else:
            # --- FIXED SAVING LOGIC ---
            authority_keypair = SolanaKeypair()
            # Convert keypair to bytes, then to a standard Python list for JSON
            key_as_list = list(bytes(authority_keypair))
            
            with open("authority_keypair.json", "w") as f:
                f.write(json.dumps(key_as_list))
            print(f"Generated NEW Solana Wallet: {authority_keypair.pubkey()}")

        # Check Balance
        print("Checking balance...")
        balance_resp = solana_client.get_balance(authority_keypair.pubkey())
        lamports = balance_resp.value
        
        # If low balance (< 0.5 SOL), request airdrop
        if lamports < 500_000_000:
            print(f"Balance low ({lamports} lamports). Requesting Devnet Airdrop...")
            try:
                # 1 SOL = 1,000,000,000 Lamports
                solana_client.request_airdrop(authority_keypair.pubkey(), 1_000_000_000)
                time.sleep(2) 
                print("Airdrop requested successfully.")
            except Exception as e:
                print(f"Airdrop failed (might be rate limited): {e}")
        else:
            print(f"Wallet funded: {lamports / 1_000_000_000:.2f} SOL")

    except Exception as e:
        print(f"Failed to initialize Solana: {e}")

def _build_story(trip_uuid, pred, sim_data, mitigation_data):
    """Generate a human-readable journey story for a shipment."""
    risk = pred["risk"]
    delay = pred["expected_delay_hours"]
    current = _determine_state(risk)
    previous = "on-time"  # all shipments start on-time

    cause = _get_weather_cause(trip_uuid)
    cause_phrase = f" due to {cause}" if cause else ""

    # Build the narrative
    if current == "on-time":
        story = (
            f"Shipment {trip_uuid} is on-time with a low risk score of "
            f"{risk*100:.0f}%. No delays expected."
        )
    elif current == "at-risk":
        story = (
            f"Shipment {trip_uuid} was on-time until risk increased to "
            f"{risk*100:.0f}%{cause_phrase}. "
            f"Expected delay is {delay:.1f} hours. Monitoring closely."
        )
    else:  # delayed
        story = (
            f"Shipment {trip_uuid} was on-time until risk surged to "
            f"{risk*100:.0f}%{cause_phrase}, triggering a delayed state. "
            f"Expected delay is {delay:.1f} hours."
        )

        # Add simulation context if available
        if sim_data:
            story += (
                f" Monte Carlo analysis shows delays ranging from "
                f"{sim_data['best_case']:.1f}h (best) to "
                f"{sim_data['worst_case']:.1f}h (worst case)."
            )

        # Add mitigation context if available
        if mitigation_data:
            story += (
                f" Mitigation applied: {mitigation_data['strategy']}. "
                f"Risk reduced by {mitigation_data['expected_risk_reduction']*100:.0f}%."
            )

    return story, previous, current


def generate_stories():
    """Generate journey stories for top risky shipments, optionally store on Solana."""
    global story_cache

    if predictions_cache is None:
        print("WARNING: Predictions not available, skipping stories")
        return

    story_cache = {}

    # Sort by risk desc, generate stories for top 20 non-trivial shipments
    sorted_trips = sorted(
        predictions_cache.items(), key=lambda x: x[1]["risk"], reverse=True
    )

    # Only generate stories for at-risk or delayed shipments (risk > 0.4)
    risky_trips = [(uid, p) for uid, p in sorted_trips if p["risk"] > 0.4][:20]

    # Top 10 get Solana memos
    top_for_chain = set(uid for uid, _ in risky_trips[:10]) if SOLANA_ENABLED else set()

    print(f"Generating journey stories for {len(risky_trips)} shipments...")

    for uid, pred in risky_trips:
        sim_data = simulation_cache.get(uid) if simulation_cache else None
        mit_data = mitigation_cache.get(uid) if mitigation_cache else None

        story_text, prev_state, curr_state = _build_story(uid, pred, sim_data, mit_data)

        solana_tx = None
        if uid in top_for_chain:
            memo = f"ChainReaction|story|{uid}|{prev_state}->{curr_state}|risk:{pred['risk']:.2f}"
            solana_tx = _trigger_solana_memo(uid, pred["risk"], memo)
            time.sleep(1)

        story_cache[uid] = {
            "story": story_text,
            "previous_state": prev_state,
            "current_state": curr_state,
            "risk": pred["risk"],
            "expected_delay_hours": pred["expected_delay_hours"],
            "solana_tx": solana_tx,
        }

    on_chain = sum(1 for s in story_cache.values() if s["solana_tx"])
    print(f"Stories generated: {len(story_cache)} stories, {on_chain} on-chain")


@app.on_event("startup")
async def startup_event():
    """Load data, build graph, and load models on startup. No inference runs here."""
    print("Starting up Chain-Reaction API...")
    load_shipments()
    load_weather()
    train_models()
    setup_solana_wallet()
    print("Startup complete. Awaiting /run-analysis to execute pipeline.")


@app.get("/")
async def root():
    return {
        "message": "Chain-Reaction Supply Chain API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "graph_loaded": graph is not None,
        "shipments_loaded": shipments_df is not None,
        "weather_loaded": weather_df is not None,
        "analysis_executed": analysis_executed,
    }


@app.post("/run-analysis")
async def run_analysis():
    """Execute the full analysis pipeline: predict, simulate, mitigate, generate stories."""
    global analysis_executed

    if graph is None or shipments_df is None:
        return {"error": "Data not loaded. Cannot run analysis."}
    if risk_classifier is None or delay_regressor is None:
        return {"error": "Models not loaded. Cannot run analysis."}

    print("Running full analysis pipeline...")

    compute_predictions()
    run_simulations()
    compute_mitigations()
    generate_stories()

    analysis_executed = True

    high_risk = sum(1 for v in predictions_cache.values() if v["risk"] > 0.7) if predictions_cache else 0
    total = len(predictions_cache) if predictions_cache else 0

    print("Analysis pipeline complete.")
    return {
        "status": "complete",
        "total_trips": total,
        "high_risk_trips": high_risk,
        "simulations_run": len(simulation_cache) if simulation_cache else 0,
        "mitigations_computed": len(mitigation_cache) if mitigation_cache else 0,
        "stories_generated": len(story_cache) if story_cache else 0,
    }


@app.get("/graph/stats")
async def graph_stats():
    """Return graph statistics to verify Phase 2."""
    if graph is None:
        return {"error": "Graph not loaded"}

    # Pick a sample edge for inspection
    sample_edge = None
    edges_list = list(graph.edges(data=True))
    if edges_list:
        src, dst, attrs = edges_list[0]
        sample_edge = {"source": src, "destination": dst, **attrs}

    # Pick a sample node
    sample_node = None
    nodes_list = list(graph.nodes(data=True))
    if nodes_list:
        name, attrs = nodes_list[0]
        sample_node = {"name": name, **attrs}

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "sample_edge": sample_edge,
        "sample_node": sample_node,
        "total_shipment_records": len(shipments_df) if shipments_df is not None else 0,
    }


@app.get("/weather/stats")
async def weather_stats():
    """Return weather merge statistics to verify Phase 3."""
    if graph is None:
        return {"error": "Graph not loaded"}

    nodes_with_weather = sum(1 for _, d in graph.nodes(data=True) if "temperature" in d)
    nodes_without = graph.number_of_nodes() - nodes_with_weather

    # Sample a node that has weather data
    sample = None
    for name, attrs in graph.nodes(data=True):
        if "temperature" in attrs:
            sample = {"name": name, **{k: attrs[k] for k in ["state", "historical_delay_prob"] + WEATHER_ATTRS}}
            break

    return {
        "total_nodes": graph.number_of_nodes(),
        "nodes_with_weather": nodes_with_weather,
        "nodes_without_weather": nodes_without,
        "weather_records_loaded": len(weather_df) if weather_df is not None else 0,
        "sample_node_with_weather": sample,
    }


@app.post("/predict")
async def predict():
    """Return risk predictions for all shipments (Phase 5)."""
    if predictions_cache is None:
        return {"error": "Predictions not computed yet"}
    return predictions_cache


@app.post("/simulate")
async def simulate():
    """Return Monte Carlo simulation results for all shipments (Phase 6)."""
    if simulation_cache is None:
        return {"error": "Simulations not computed yet"}
    return simulation_cache


@app.post("/mitigate")
async def mitigate():
    """Return mitigation strategies for high-risk shipments (Phase 7)."""
    if mitigation_cache is None:
        return {"error": "Mitigations not computed yet"}
    return mitigation_cache


@app.get("/story")
async def get_stories():
    """Return journey stories for risky shipments (Phase 8)."""
    if story_cache is None:
        return {"error": "Stories not generated yet"}
    return story_cache


@app.get("/graph/viz")
async def graph_viz():
    """Return graph nodes and edges for network visualization.
    Before analysis: neutral colors showing structure only.
    After analysis: risk-colored nodes and edges.
    """
    if graph is None:
        return {"error": "Graph not loaded"}

    # Build node risk from predictions (only if analysis has run)
    node_risk = {}
    if analysis_executed and predictions_cache and shipments_df is not None:
        df = shipments_df.copy()
        df["risk"] = df["trip_uuid"].map(
            lambda uid: predictions_cache.get(uid, {}).get("risk", 0)
        )
        for col in ["source_name", "destination_name"]:
            agg = df.groupby(col)["risk"].mean()
            for name, risk in agg.items():
                node_risk[name] = max(node_risk.get(name, 0), float(risk))

    # Determine node type from name patterns
    def _node_type(name):
        lower = name.lower()
        if "port" in lower or "harbour" in lower:
            return "port"
        if "warehouse" in lower or "wh" in lower or "distribution" in lower:
            return "warehouse"
        if "hub" in lower or "central" in lower:
            return "hub"
        return "supplier"

    nodes = []
    for name in graph.nodes:
        state = graph.nodes[name].get("state", "on-time")
        risk = node_risk.get(name, 0)
        if analysis_executed:
            color = "#ff3864" if risk > 0.7 else "#ffd700" if risk > 0.4 else "#00ff88"
        else:
            color = "#3b82f6"  # neutral blue before analysis

        nodes.append({
            "id": name,
            "label": name.split("(")[0].strip().split("_")[0],
            "risk": round(risk, 4),
            "state": state,
            "type": _node_type(name),
            "color": color,
        })

    edges = []
    for src, dst, attrs in graph.edges(data=True):
        if analysis_executed:
            src_risk = node_risk.get(src, 0)
            dst_risk = node_risk.get(dst, 0)
            edge_risk = max(src_risk, dst_risk)
            edge_color = "#ff3864" if edge_risk > 0.7 else "#ffd700" if edge_risk > 0.4 else "#00ff88"
        else:
            edge_color = "rgba(59,130,246,0.4)"  # neutral blue before analysis

        edges.append({
            "source": src,
            "target": dst,
            "transit_time": attrs.get("transit_time", 0),
            "distance": attrs.get("distance", 0),
            "color": edge_color,
        })

    return {"nodes": nodes, "edges": edges, "analysis_executed": analysis_executed}


@app.get("/risk/stats")
async def risk_stats():
    """Return risk model statistics to verify Phase 4."""
    if predictions_cache is None:
        return {"error": "Predictions not computed"}

    risks = [v["risk"] for v in predictions_cache.values()]
    delays = [v["expected_delay_hours"] for v in predictions_cache.values()]

    # Sample top-5 riskiest trips
    sorted_trips = sorted(predictions_cache.items(), key=lambda x: x[1]["risk"], reverse=True)
    top_5 = {uid: pred for uid, pred in sorted_trips[:5]}

    return {
        "total_trips": len(predictions_cache),
        "risk_mean": round(float(np.mean(risks)), 4),
        "risk_max": round(float(np.max(risks)), 4),
        "risk_min": round(float(np.min(risks)), 4),
        "delay_mean_hours": round(float(np.mean(delays)), 2),
        "delay_max_hours": round(float(np.max(delays)), 2),
        "high_risk_trips": sum(1 for r in risks if r > 0.7),
        "models_loaded": risk_classifier is not None and delay_regressor is not None,
        "top_5_riskiest": top_5,
    }


@app.get("/backtest")
async def backtest_data():
    """Return actual vs predicted delay for model validation visualization."""
    if delay_regressor is None or shipments_df is None:
        return {"error": "Model or data not loaded"}

    df, _ = _build_feature_df()
    # Ensure consistent time-based split with training
    split_cutoff = None
    if model_meta and "split_cutoff_ts" in model_meta:
        split_cutoff = model_meta["split_cutoff_ts"]

    df = df.sort_values("od_start_time_parsed").dropna(subset=["od_start_time_parsed"])
    if split_cutoff is None:
        _, test, split_cutoff = _time_based_split(df, test_fraction=0.3)
    else:
        test = df[df["od_start_time_parsed"] >= split_cutoff].copy()

    X_test = test[FEATURE_COLS]
    pred = delay_regressor.predict(X_test)
    test["predicted_delay"] = np.clip(pred, 0, None)

    len0_test = len(test)

    # Downsample to ~150 points for rendering
    n = min(150, len(test))
    if len(test) > n:
        sample_idx = np.linspace(0, len(test) - 1, n).astype(int)
        test = test.iloc[sample_idx]

    points = []
    for _, row in test.iterrows():
        points.append({
            "ts": row["od_start_time_parsed"].strftime("%b %d"),
            "actual": round(float(row["delay_hours"]), 2),
            "predicted": round(float(row["predicted_delay"]), 2),
        })

    actual_arr = test["delay_hours"].values
    pred_arr = test["predicted_delay"].values
    err = actual_arr - pred_arr
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    return {
        "points": points,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "test_size": len0_test,
        "train_size": len(df) - len0_test,
    }
