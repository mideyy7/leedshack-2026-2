from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
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


@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    print("Starting up Chain-Reaction API...")
    load_shipments()


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
