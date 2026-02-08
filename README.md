# Chain-Reaction: Predictive Digital Twin for Global Supply Chains

A hackathon MVP that combines multivariate forecasting with a supply-chain digital twin. The goal is to turn historical logistics + weather data into predictive signals and actionable mitigation for operations teams.

## Challenge Alignment (Parallax + Ligentia)

- **Parallax: Use the Past to Predict the Future**
This project performs multivariate forecasting across shipment, route, and weather datasets to predict delays and risk. It focuses on correlations between multiple datasets and uses a learnable model to forecast outcomes from historical signals.

- **Ligentia: The Great Supply Chain Race**
Chain-Reaction provides early warning signals, simulation, and a shipment story view to move teams from reactive to predictive decision-making. It models a supply-chain graph, highlights risk at node and route level, and suggests mitigation paths.

## What It Does

- Builds a directed supply-chain graph from shipment history
- Merges weather features onto nodes and shipment records
- Trains LightGBM models for risk classification and delay regression
- Backend: endpoints for prediction, simulation, mitigation, and storytelling
- Frontend: a React dashboard for exploration and visualization

## Tech Stack

- **Frontend**: React, Recharts, Axios
- **Backend**: FastAPI, Python
- **ML/Data**: Pandas, NetworkX, NumPy, LightGBM, Scikit-learn
- **Blockchain**: Solana devnet memo transactions for high-risk mitigation audit trail.

## Quick Start

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`.

### 2. Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`.

### 3. Data

Place CSVs in `data/`:

- `delivery.csv` - shipment history
- `weather2.csv` - weather history used for feature joins

## API Endpoints (Backend)

- `GET /` - API info
- `GET /health` - health check
- `POST /run-analysis` - build graph, merge weather, train models
- `GET /graph/stats` - graph summary
- `GET /graph/viz` - graph nodes/edges for visualization
- `GET /weather/stats` - weather summary
- `POST /predict` - risk + expected delay for shipments
- `POST /simulate` - Monte Carlo delay simulation
- `POST /mitigate` - suggested mitigation (optionally writes Solana memo)
- `GET /story` - shipment journey narrative
- `GET /risk/stats` - aggregate risk metrics
- `GET /backtest` - historical accuracy snapshot

## How It Maps to the Challenges

- **Parallax**: Multivariate forecasting across logistics + weather signals
- **Ligentia**: Predictive supply-chain monitoring, early warning, and journey storytelling

## License

MIT License - Hackathon Project
