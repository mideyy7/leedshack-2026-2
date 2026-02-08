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
story_cache = None        # dict of trip_uuid -> {story, previous_state, current_state, risk, ...}
analysis_executed = False  # True after /run-analysis has been called

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
    # Basic cleanup
    if match:
        return match.group(1).strip()
    return "Unknown Region"


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
    df["is_delayed"] = (df["factor"] > 1.5).astype(int)
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

    risk_classifier = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        num_leaves=31, verbose=-1, n_jobs=-1,
    )
    risk_classifier.fit(X, y_cls)

    delay_regressor = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        num_leaves=31, verbose=-1, n_jobs=-1,
    )
    delay_regressor.fit(X, y_reg)

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


N_SIMULATIONS = 100


def run_simulations():
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
            noise_scale = 0.2 * (1 + weather_severity)
            noise = rng.normal(1.0, noise_scale, size=len(df))
            X_sim[:, ci] = X_base[:, ci] * noise

        preds = delay_regressor.predict(X_sim)
        all_delays[:, sim] = np.clip(preds, 0, None)

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

    print(f"Monte Carlo simulations done for {len(simulation_cache)} trips")


MITIGATION_STRATEGIES = [
    {"strategy": "Reroute shipment via alternate corridor", "expected_risk_reduction": 0.25},
    {"strategy": "Expedite critical leg with express carrier", "expected_risk_reduction": 0.20},
    {"strategy": "Hold inventory upstream at distribution center", "expected_risk_reduction": 0.15},
]


def _select_strategy(risk):
    if risk > 0.9:
        return MITIGATION_STRATEGIES[0]
    elif risk > 0.8:
        return MITIGATION_STRATEGIES[1]
    else:
        return MITIGATION_STRATEGIES[2]


def _trigger_solana_memo(trip_uuid, risk, action_desc):
    import time
    try:
        client = SolanaClient(SOLANA_RPC_URL)
        keypair = SolanaKeypair()

        memo_text = f"ChainReaction|{trip_uuid}|risk:{risk:.2f}|action:{action_desc}"
        memo_bytes = memo_text.encode("utf-8")[:566]

        ix = Instruction(
            program_id=MEMO_PROGRAM_ID,
            accounts=[AccountMeta(pubkey=keypair.pubkey(), is_signer=True, is_writable=True)],
            data=memo_bytes,
        )

        airdrop_resp = client.request_airdrop(keypair.pubkey(), 1_000_000_000)
        time.sleep(3)

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
        import hashlib
        mock_sig = hashlib.sha256(f"ChainReaction:{trip_uuid}:{risk}".encode()).hexdigest()[:88]
        return f"DEMO_{mock_sig}"


def compute_mitigations():
    global mitigation_cache

    if predictions_cache is None:
        print("WARNING: Predictions not available, skipping mitigations")
        return

    mitigation_cache = {}
    high_risk = {uid: p for uid, p in predictions_cache.items() if p["risk"] > HIGH_RISK_THRESHOLD}

    print(f"Computing mitigations for {len(high_risk)} high-risk shipments...")

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

    print(f"Mitigations computed: {len(mitigation_cache)} strategies")


def _determine_state(risk):
    if risk > 0.7:
        return "delayed"
    elif risk > 0.4:
        return "at-risk"
    return "on-time"


def _get_weather_cause(trip_uuid):
    if shipments_df is None or graph is None:
        return None

    rows = shipments_df[shipments_df["trip_uuid"] == trip_uuid]
    if rows.empty:
        return None

    source = rows.iloc[0]["source_name"]
    attrs = graph.nodes.get(source, {})

    rain = float(attrs.get("rain_1h", 0))
    snow = float(attrs.get("snow_1h", 0))
    wind = float(attrs.get("wind_speed", 0))
    weather = attrs.get("weather_main", "Clear")

    if snow > 0: return f"snowfall ({weather})"
    if rain > 0.5: return f"heavy rain ({weather})"
    if wind > 10: return f"high winds ({weather})"
    if rain > 0: return f"rain ({weather})"
    if weather not in ("Clear", "Clouds"): return weather.lower()
    return None


def _build_story(trip_uuid, pred, sim_data, mitigation_data):
    risk = pred["risk"]
    delay = pred["expected_delay_hours"]
    current = _determine_state(risk)
    previous = "on-time"

    cause = _get_weather_cause(trip_uuid)
    cause_phrase = f" due to {cause}" if cause else ""

    if current == "on-time":
        story = f"Shipment {trip_uuid} is on-time with a low risk score of {risk*100:.0f}%. No delays expected."
    elif current == "at-risk":
        story = f"Shipment {trip_uuid} was on-time until risk increased to {risk*100:.0f}%{cause_phrase}. Expected delay is {delay:.1f} hours."
    else:
        story = f"Shipment {trip_uuid} was on-time until risk surged to {risk*100:.0f}%{cause_phrase}, triggering a delayed state. Expected delay is {delay:.1f} hours."

    if sim_data:
        story += f" Monte Carlo: {sim_data['best_case']:.1f}h (best) to {sim_data['worst_case']:.1f}h (worst)."

    if mitigation_data:
        story += f" Mitigation applied: {mitigation_data['strategy']}."

    return story, previous, current


def generate_stories():
    global story_cache

    if predictions_cache is None:
        print("WARNING: Predictions not available, skipping stories")
        return

    story_cache = {}
    sorted_trips = sorted(predictions_cache.items(), key=lambda x: x[1]["risk"], reverse=True)
    risky_trips = [(uid, p) for uid, p in sorted_trips if p["risk"] > 0.4][:20]
    top_for_chain = set(uid for uid, _ in risky_trips[:3]) if SOLANA_ENABLED else set()

    print(f"Generating stories for {len(risky_trips)} shipments...")

    for uid, pred in risky_trips:
        sim_data = simulation_cache.get(uid) if simulation_cache else None
        mit_data = mitigation_cache.get(uid) if mitigation_cache else None

        story_text, prev_state, curr_state = _build_story(uid, pred, sim_data, mit_data)

        solana_tx = None
        if uid in top_for_chain:
            memo = f"ChainReaction|story|{uid}|{prev_state}->{curr_state}|risk:{pred['risk']:.2f}"
            solana_tx = _trigger_solana_memo(uid, pred["risk"], memo)

        story_cache[uid] = {
            "story": story_text,
            "previous_state": prev_state,
            "current_state": curr_state,
            "risk": pred["risk"],
            "expected_delay_hours": pred["expected_delay_hours"],
            "solana_tx": solana_tx,
        }

    print(f"Stories generated: {len(story_cache)}")


@app.on_event("startup")
async def startup_event():
    print("Starting up Chain-Reaction API...")
    load_shipments()
    load_weather()
    train_models()
    print("Startup complete. Awaiting /run-analysis.")


@app.get("/")
async def root():
    return {"message": "Chain-Reaction Supply Chain API", "version": "1.0.0"}


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
    global analysis_executed

    if graph is None or shipments_df is None or risk_classifier is None:
        return {"error": "System not ready. Data or models missing."}

    print("Running full analysis pipeline...")
    compute_predictions()
    run_simulations()
    compute_mitigations()
    generate_stories()

    analysis_executed = True
    print("Analysis pipeline complete.")
    
    total = len(predictions_cache) if predictions_cache else 0
    high_risk = sum(1 for v in predictions_cache.values() if v["risk"] > 0.7) if predictions_cache else 0

    return {
        "status": "complete",
        "total_trips": total,
        "high_risk_trips": high_risk,
        "simulations_run": len(simulation_cache) if simulation_cache else 0,
        "mitigations_computed": len(mitigation_cache) if mitigation_cache else 0,
        "stories_generated": len(story_cache) if story_cache else 0,
    }


@app.get("/graph/viz")
async def graph_viz():
    """
    Return aggregated graph nodes and edges grouped by STATE.
    Aggregates risk from individual facilities up to the state level.
    """
    if graph is None:
        return {"error": "Graph not loaded"}

    # 1. Map Shipments to Risk (if analysis executed)
    # We create a lookup of risk by trip_uuid
    trip_risk_lookup = {}
    if analysis_executed and predictions_cache:
        for uid, data in predictions_cache.items():
            trip_risk_lookup[uid] = data.get("risk", 0.0)

    # 2. Map Nodes (Facilities) to Risk
    # We iterate through shipments_df to find which nodes are associated with which trips
    node_risks = {}  # {node_name: [list of risks]}
    
    if analysis_executed and shipments_df is not None:
        # Create a quick lookup for risk per shipment row
        # This is faster than iterating rows if dataframe is large
        shipments_df['temp_risk'] = shipments_df['trip_uuid'].map(trip_risk_lookup).fillna(0.0)
        
        # Group by Source and Dest to collect risks
        src_risks = shipments_df.groupby("source_name")['temp_risk'].apply(list).to_dict()
        dst_risks = shipments_df.groupby("destination_name")['temp_risk'].apply(list).to_dict()
        
        # Merge source and dest risks into node_risks
        for node in graph.nodes:
            risks = src_risks.get(node, []) + dst_risks.get(node, [])
            if risks:
                node_risks[node] = risks

    # 3. Aggregate by STATE
    state_nodes = {}  # {state_name: {risk_sum, count, type}}
    
    for node in graph.nodes:
        state = _extract_state(node)
        if not state or state == "Unknown Region": 
            continue # Skip nodes without clear state mapping for the high-level view

        if state not in state_nodes:
            state_nodes[state] = {"risk_values": [], "facility_count": 0}
        
        state_nodes[state]["facility_count"] += 1
        
        # If we have risks for this specific facility, add them to the state bucket
        if node in node_risks:
            state_nodes[state]["risk_values"].extend(node_risks[node])

    # 4. Build Final Node List
    final_nodes = []
    for state, data in state_nodes.items():
        # Calculate average risk for the state
        avg_risk = 0.0
        if data["risk_values"]:
            avg_risk = sum(data["risk_values"]) / len(data["risk_values"])
        
        # Determine color based on risk
        color = "#3b82f6" # default blue
        if analysis_executed:
            if avg_risk > 0.7: color = "#ff3864"
            elif avg_risk > 0.4: color = "#ffd700"
            else: color = "#00ff88"

        final_nodes.append({
            "id": state,
            "label": state,
            "risk": round(avg_risk, 4),
            "type": "state_hub",
            "color": color
        })

    # 5. Build Aggregated Edges
    # We iterate existing edges and map them to State -> State edges
    state_edges = {} # {(state_a, state_b): {count}}

    for u, v in graph.edges:
        state_u = _extract_state(u)
        state_v = _extract_state(v)

        if state_u and state_v and state_u != state_v and state_u != "Unknown Region" and state_v != "Unknown Region":
            key = (state_u, state_v)
            if key not in state_edges:
                state_edges[key] = 0
            state_edges[key] += 1

    final_edges = []
    for (src, dst), count in state_edges.items():
        final_edges.append({
            "source": src,
            "target": dst,
            "weight": count,
            "color": "rgba(139, 146, 168, 0.4)" # default grey
        })

    return {"nodes": final_nodes, "edges": final_edges, "analysis_executed": analysis_executed}


@app.post("/predict")
async def predict():
    if predictions_cache is None: return {"error": "Predictions not computed"}
    return predictions_cache

@app.post("/simulate")
async def simulate():
    if simulation_cache is None: return {"error": "Simulations not computed"}
    return simulation_cache

@app.post("/mitigate")
async def mitigate():
    if mitigation_cache is None: return {"error": "Mitigations not computed"}
    return mitigation_cache

@app.get("/story")
async def get_stories():
    if story_cache is None: return {"error": "Stories not generated"}
    return story_cache