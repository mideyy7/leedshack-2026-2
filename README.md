# Chain-Reaction: Predictive Digital Twin for Global Supply Chains

A hackathon MVP that builds a predictive digital twin for global supply chains using real-time data analysis, weather integration, and risk prediction.

## Tech Stack

- **Frontend**: React, Recharts, Axios
- **Backend**: FastAPI, Python
- **Data Processing**: Pandas, NetworkX, Scikit-learn

## Project Structure

```
leeds-hack/
├── backend/          # FastAPI backend
│   ├── main.py       # API endpoints
│   ├── requirements.txt
│   └── README.md
├── frontend/         # React frontend
│   ├── src/
│   ├── package.json
│   └── README.md
├── data/             # CSV datasets
│   ├── delivery.csv
│   └── weather.csv
└── README.md
```

## Quick Start

### 1. Setup Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend will run at: http://localhost:8000

### 2. Setup Frontend

```bash
cd frontend
npm install
npm start
```

Frontend will run at: http://localhost:3000

### 3. Add Data

Place your CSV files in the `data/` folder:
- `delivery.csv` - Shipment delivery data
- `weather.csv` - Weather data

## Development Phases

- [x] **Phase 1**: Project Setup
- [x] **Phase 2**: Load Shipment Data & Build Graph
- [x] **Phase 3**: Merge Weather Data
- [x] **Phase 4**: Multivariate Risk Prediction (LightGBM)
- [x] **Phase 5**: /predict API Endpoint + Dashboard
- [x] **Phase 6**: Monte Carlo /simulate Endpoint
- [x] **Phase 7**: /mitigate Endpoint + Solana Integration
- [ ] **Phase 8**: Journey Story
- [ ] **Phase 9**: Dashboard Visualization

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Get risk predictions (Phase 5)
- `POST /simulate` - Monte Carlo simulation (Phase 6)
- `POST /mitigate` - Mitigation strategies (Phase 7)

## Features

- Real-time supply chain monitoring
- Weather-based risk prediction
- Monte Carlo simulation for delay scenarios
- Automated mitigation strategy suggestions
- Interactive visualization dashboard

## License

MIT License - Hackathon Project
