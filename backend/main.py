from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
import numpy as np
import lightgbm as lgb
import joblib
import re
import os
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
risk_classifier = None
delay_regressor = None
predictions_cache = None  # dict of trip_uuid -> {risk, expected_delay_hours}
simulation_cache = None   # dict of trip_uuid -> {best_case, worst_case, expected_delay_hours, p10, p90}
mitigation_cache = None   # dict of trip_uuid -> {strategy, expected_risk_reduction, solana_tx}

# Solana config
SOLANA_RPC_URL = "https://api.devnet.solana.com"
MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
HIGH_RISK_THRESHOLD = 0.7
SOLANA_ENABLED = os.environ.get("SOLANA_ENABLED", "true").lower() == "true"

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
    """Extract state from node like 'Agra_Central_D_3 (Uttar Pradesh)' -> 'Uttar Pradesh'."""
    match = re.search(r"\(([^)]+)\)", node_name)
    return match.group(1) if match else None


WEATHER_ATTRS = ["temperature", "humidity", "wind_speed", "rain_1h", "snow_1h", "weather_main"]


def load_weather():
    """Load weather CSV and merge weather attributes onto graph nodes."""
    global weather_df

    csv_path = os.path.join(DATA_DIR, "weather.csv")
    if not os.path.exists(csv_path):
        print("WARNING: weather.csv not found at", csv_path)
        return

    weather_df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(weather_df)} weather records")

    if graph is None:
        print("WARNING: Graph not built yet, skipping weather merge")
        return

    # Filter to Indian weather stations (shipments are India-based)
    india_weather = weather_df[weather_df["country"] == "IN"].copy()
    india_weather["name_lower"] = india_weather["name"].str.lower()

    # Build lookup: lowercase city name -> average weather row
    city_weather = india_weather.groupby("name_lower").agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"),
        rain_1h=("rain_1h", "mean"),
        snow_1h=("snow_1h", "mean"),
        weather_main=("weather_main", "first"),
    )

    # Build state-level fallback averages
    node_states = {}
    for node in graph.nodes:
        state = _extract_state(node)
        if state:
            node_states.setdefault(state, []).append(node)

    # Country-wide fallback
    india_avg = {
        "temperature": float(india_weather["temperature"].mean()),
        "humidity": float(india_weather["humidity"].mean()),
        "wind_speed": float(india_weather["wind_speed"].mean()),
        "rain_1h": float(india_weather["rain_1h"].mean()),
        "snow_1h": float(india_weather["snow_1h"].mean()),
        "weather_main": "Clouds",
    }

    matched_exact, matched_state, matched_fallback = 0, 0, 0

    for node in graph.nodes:
        city = _extract_city(node)

        # Strategy 1: exact city name match
        if city in city_weather.index:
            row = city_weather.loc[city]
            for attr in WEATHER_ATTRS:
                val = row[attr]
                graph.nodes[node][attr] = round(float(val), 2) if attr != "weather_main" else str(val)
            matched_exact += 1
            continue

        # Strategy 2: pick any matched node from same state
        state = _extract_state(node)
        assigned = False
        if state and state in node_states:
            for sibling in node_states[state]:
                sib_city = _extract_city(sibling)
                if sib_city in city_weather.index:
                    row = city_weather.loc[sib_city]
                    for attr in WEATHER_ATTRS:
                        val = row[attr]
                        graph.nodes[node][attr] = round(float(val), 2) if attr != "weather_main" else str(val)
                    assigned = True
                    matched_state += 1
                    break

        if assigned:
            continue

        # Strategy 3: India-wide average fallback
        for attr in WEATHER_ATTRS:
            graph.nodes[node][attr] = india_avg[attr] if attr != "weather_main" else india_avg["weather_main"]
        matched_fallback += 1

    print(f"Weather merged: {matched_exact} exact, {matched_state} state-level, {matched_fallback} fallback")


# --- Phase 4: Feature engineering & LightGBM training ---

FEATURE_COLS = [
    "osrm_time", "actual_distance_to_destination", "segment_osrm_time",
    "segment_osrm_distance", "factor", "segment_factor",
    "temperature", "humidity", "wind_speed", "rain_1h", "snow_1h",
    "route_type_encoded", "weather_main_encoded",
    "hour_of_day", "day_of_week", "is_peak_hour",
]


def _build_feature_df():
    """Build a feature DataFrame by merging shipment rows with graph weather attributes."""
    df = shipments_df.copy()

    # Parse temporal features from od_start_time
    df["od_start_time_parsed"] = pd.to_datetime(df["od_start_time"], format="mixed", dayfirst=False, errors="coerce")
    df["hour_of_day"] = df["od_start_time_parsed"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["od_start_time_parsed"].dt.dayofweek.fillna(0).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].apply(lambda h: 1 if 7 <= h <= 10 or 16 <= h <= 20 else 0)

    # Encode categoricals
    route_map = {rt: i for i, rt in enumerate(df["route_type"].dropna().unique())}
    df["route_type_encoded"] = df["route_type"].map(route_map).fillna(0).astype(int)

    # Merge weather from source node
    weather_lookup = {}
    if graph is not None:
        for node, attrs in graph.nodes(data=True):
            if "temperature" in attrs:
                weather_lookup[node] = {k: attrs[k] for k in ["temperature", "humidity", "wind_speed", "rain_1h", "snow_1h", "weather_main"]}

    weather_main_categories = set()
    for w in weather_lookup.values():
        weather_main_categories.add(w["weather_main"])

    weather_main_map = {wm: i for i, wm in enumerate(sorted(weather_main_categories))}

    for col in ["temperature", "humidity", "wind_speed", "rain_1h", "snow_1h"]:
        df[col] = df["source_name"].map(lambda n, c=col: weather_lookup.get(n, {}).get(c, 0.0))

    df["weather_main_raw"] = df["source_name"].map(lambda n: weather_lookup.get(n, {}).get("weather_main", "Clouds"))
    df["weather_main_encoded"] = df["weather_main_raw"].map(weather_main_map).fillna(0).astype(int)

    # Define targets
    # Classification: delayed if factor > 1.5 (actual took 50%+ longer than OSRM estimate)
    df["is_delayed"] = (df["factor"] > 1.5).astype(int)

    # Regression: delay in hours = (actual_time - osrm_time) / 60, clipped to >= 0
    df["delay_hours"] = ((df["actual_time"] - df["osrm_time"]) / 60).clip(lower=0)

    # Fill NaN in feature columns
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df, weather_main_map


def train_models():
    """Train LightGBM classifier and regressor, save to disk."""
    global risk_classifier, delay_regressor

    os.makedirs(MODEL_DIR, exist_ok=True)
    clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
    reg_path = os.path.join(MODEL_DIR, "regressor.pkl")
    meta_path = os.path.join(MODEL_DIR, "meta.pkl")

    # If models already exist, load them
    if os.path.exists(clf_path) and os.path.exists(reg_path):
        risk_classifier = joblib.load(clf_path)
        delay_regressor = joblib.load(reg_path)
        print("Loaded existing models from disk")
        return

    print("Training LightGBM models...")
    df, weather_main_map = _build_feature_df()

    X = df[FEATURE_COLS].values
    y_cls = df["is_delayed"].values
    y_reg = df["delay_hours"].values

    # Train classifier
    risk_classifier = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        num_leaves=31, verbose=-1, n_jobs=-1,
    )
    risk_classifier.fit(X, y_cls)

    # Train regressor
    delay_regressor = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        num_leaves=31, verbose=-1, n_jobs=-1,
    )
    delay_regressor.fit(X, y_reg)

    # Save models and metadata
    joblib.dump(risk_classifier, clf_path)
    joblib.dump(delay_regressor, reg_path)
    joblib.dump({"weather_main_map": weather_main_map}, meta_path)

    print(f"Models trained and saved. Delay rate: {y_cls.mean():.2%}, Mean delay: {y_reg.mean():.1f}h")


def compute_predictions():
    """Run predictions for all shipments and cache results."""
    global predictions_cache

    if risk_classifier is None or delay_regressor is None:
        print("WARNING: Models not loaded, skipping predictions")
        return

    df, _ = _build_feature_df()
    X = df[FEATURE_COLS].values

    risk_probs = risk_classifier.predict_proba(X)[:, 1]
    delay_hours = delay_regressor.predict(X)

    # Aggregate per trip_uuid (take max risk and max delay across segments)
    df["risk"] = risk_probs
    df["predicted_delay_hours"] = np.clip(delay_hours, 0, None)

    trip_preds = df.groupby("trip_uuid").agg(
        risk=("risk", "max"),
        expected_delay_hours=("predicted_delay_hours", "max"),
    )

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

    # Columns we'll perturb per simulation
    perturb_cols = ["osrm_time", "segment_osrm_time", "temperature", "humidity",
                    "wind_speed", "rain_1h", "snow_1h"]

    X_base = df[FEATURE_COLS].values.copy()

    # Weather severity multiplier: heavier rain/snow/wind → more variance
    weather_severity = (
        df["rain_1h"].values * 0.3
        + df["snow_1h"].values * 0.5
        + df["wind_speed"].values * 0.02
    ).clip(0, 1)  # 0-1 scale

    # Run N simulations, collect delay predictions for each row
    all_delays = np.zeros((len(df), N_SIMULATIONS))

    col_indices = [FEATURE_COLS.index(c) for c in perturb_cols]

    for sim in range(N_SIMULATIONS):
        X_sim = X_base.copy()
        for ci in col_indices:
            # Perturb each feature by ±20%, scaled by weather severity
            noise_scale = 0.2 * (1 + weather_severity)
            noise = rng.normal(1.0, noise_scale, size=len(df))
            X_sim[:, ci] = X_base[:, ci] * noise

        preds = delay_regressor.predict(X_sim)
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


def _select_strategy(risk):
    """Pick a mitigation strategy based on risk severity."""
    if risk > 0.9:
        return MITIGATION_STRATEGIES[0]  # reroute for very high risk
    elif risk > 0.8:
        return MITIGATION_STRATEGIES[1]  # expedite for high risk
    else:
        return MITIGATION_STRATEGIES[2]  # hold upstream for moderate-high


def _trigger_solana_memo(trip_uuid, risk, action_desc):
    """Send a memo transaction to Solana devnet for audit trail. Returns tx signature or None."""
    import time
    try:
        client = SolanaClient(SOLANA_RPC_URL)
        keypair = SolanaKeypair()

        memo_text = f"ChainReaction|{trip_uuid}|risk:{risk:.2f}|action:{action_desc}"
        memo_bytes = memo_text.encode("utf-8")[:566]  # memo max ~566 bytes

        ix = Instruction(
            program_id=MEMO_PROGRAM_ID,
            accounts=[AccountMeta(pubkey=keypair.pubkey(), is_signer=True, is_writable=True)],
            data=memo_bytes,
        )

        # Airdrop SOL for tx fee
        airdrop_resp = client.request_airdrop(keypair.pubkey(), 1_000_000_000)  # 1 SOL
        # Wait for airdrop confirmation
        time.sleep(3)

        # Get a recent blockhash and build transaction
        resp = client.get_latest_blockhash()
        blockhash = resp.value.blockhash

        msg = SolanaMessage.new_with_blockhash(
            [ix], keypair.pubkey(), blockhash
        )
        tx = SolanaTransaction.new(
            [keypair], msg, blockhash
        )

        result = client.send_raw_transaction(bytes(tx))
        sig = str(result.value)
        print(f"Solana tx for {trip_uuid}: {sig}")
        return sig
    except Exception as e:
        print(f"Solana tx failed for {trip_uuid}: {e}")
        # Demo fallback: generate a deterministic mock signature for hackathon presentation
        import hashlib
        mock_sig = hashlib.sha256(f"ChainReaction:{trip_uuid}:{risk}".encode()).hexdigest()[:88]
        print(f"Using demo signature for {trip_uuid}")
        return f"DEMO_{mock_sig}"


def compute_mitigations():
    """Generate mitigation strategies for high-risk shipments, optionally trigger Solana."""
    global mitigation_cache

    if predictions_cache is None:
        print("WARNING: Predictions not available, skipping mitigations")
        return

    mitigation_cache = {}
    high_risk = {uid: p for uid, p in predictions_cache.items() if p["risk"] > HIGH_RISK_THRESHOLD}

    print(f"Computing mitigations for {len(high_risk)} high-risk shipments...")

    # Only send Solana txs for top 3 riskiest (demo budget — devnet airdrop is rate-limited)
    sorted_high = sorted(high_risk.items(), key=lambda x: x[1]["risk"], reverse=True)
    top_for_chain = set(uid for uid, _ in sorted_high[:3]) if SOLANA_ENABLED else set()

    for uid, pred in high_risk.items():
        strat = _select_strategy(pred["risk"])

        solana_tx = None
        if uid in top_for_chain:
            solana_tx = _trigger_solana_memo(uid, pred["risk"], strat["strategy"])

        mitigation_cache[uid] = {
            "strategy": strat["strategy"],
            "expected_risk_reduction": strat["expected_risk_reduction"],
            "mitigated_risk": round(max(0, pred["risk"] - strat["expected_risk_reduction"]), 4),
            "solana_tx": solana_tx,
        }

    on_chain_count = sum(1 for m in mitigation_cache.values() if m["solana_tx"])
    print(f"Mitigations computed: {len(mitigation_cache)} strategies, {on_chain_count} on-chain txs")


@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    print("Starting up Chain-Reaction API...")
    load_shipments()
    load_weather()
    train_models()
    compute_predictions()
    run_simulations()
    compute_mitigations()


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
