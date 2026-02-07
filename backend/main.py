from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
import numpy as np
import re
import os

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


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


@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    print("Starting up Chain-Reaction API...")
    load_shipments()
    load_weather()


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
