import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { BarChart, Bar, LineChart, Line, Legend, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';
const SOLANA_EXPLORER = 'https://explorer.solana.com/tx';
const CARGO_COST_PER_HOUR = 1200;

const PIPELINE_STEPS = [
  { key: 'features', label: 'Constructing feature matrix from shipment & weather data' },
  { key: 'lgbm', label: 'Running LightGBM risk classifier & delay regressor' },
  { key: 'montecarlo', label: 'Executing Monte Carlo simulation (100 scenarios)' },
  { key: 'mitigation', label: 'Computing autonomous mitigation strategies' },
  { key: 'solana', label: 'Recording mitigation decisions on Solana devnet' },
];

const formatDelay = (hours) => {
  if (hours >= 24) return `${hours.toFixed(1)}h (~${(hours / 24).toFixed(1)} days)`;
  return `${hours.toFixed(1)}h`;
};

const readableId = (uuid) => {
  const digits = uuid.replace(/\D/g, '');
  return `trip-${digits.slice(-5).padStart(5, '0')}`;
};

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [apiStatus, setApiStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(0);

  const [predictions, setPredictions] = useState(null);
  const [simulations, setSimulations] = useState(null);
  const [mitigations, setMitigations] = useState(null);
  const [stories, setStories] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [backtestData, setBacktestData] = useState(null);

  const [analysisRun, setAnalysisRun] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineSteps, setPipelineSteps] = useState([]);
  const [pipelineError, setPipelineError] = useState(null);
  const [pipelineSummary, setPipelineSummary] = useState(null);
  const [pipelineCompletedAt, setPipelineCompletedAt] = useState(null);

  const [dateRange, setDateRange] = useState(null);
  const [retrainStart, setRetrainStart] = useState('');
  const [retrainEnd, setRetrainEnd] = useState('');
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [retrainMeta, setRetrainMeta] = useState(null);

  useEffect(() => {
    const interval = setInterval(() => setLastUpdate(prev => prev + 3), 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => { checkApiHealth(); }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setApiStatus(response.data);
      if (response.data.analysis_executed) setAnalysisRun(true);
      setLoading(false);
    } catch (error) {
      setApiStatus({ status: 'error', message: 'Cannot connect to backend' });
      setLoading(false);
    }
  };

  const fetchPredictions = useCallback(async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`);
      if (!response.data.error) {
        const arr = Object.entries(response.data).map(([trip_uuid, pred]) => ({
          trip_uuid, risk: pred.risk, expected_delay_hours: pred.expected_delay_hours,
        }));
        setPredictions(arr);
      }
    } catch (error) { console.error('Failed to fetch predictions:', error); }
  }, []);

  const fetchSimulations = useCallback(async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/simulate`);
      if (!response.data.error) {
        const arr = Object.entries(response.data).map(([trip_uuid, sim]) => ({
          trip_uuid, ...sim,
        }));
        setSimulations(arr);
      }
    } catch (error) { console.error('Failed to fetch simulations:', error); }
  }, []);

  const fetchMitigations = useCallback(async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/mitigate`);
      if (!response.data.error) setMitigations(response.data);
    } catch (error) { console.error('Failed to fetch mitigations:', error); }
  }, []);

  const fetchStories = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/story`);
      if (!response.data.error) setStories(response.data);
    } catch (error) { console.error('Failed to fetch stories:', error); }
  }, []);

  const fetchGraphData = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/graph/viz`);
      if (!response.data.error) setGraphData(response.data);
    } catch (error) { console.error('Failed to fetch graph data:', error); }
  }, []);

  const fetchBacktestData = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/backtest`);
      if (!response.data.error) setBacktestData(response.data);
    } catch (error) { console.error('Failed to fetch backtest:', error); }
  }, []);

  const fetchDateRange = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/data/date-range`);
      if (!response.data.error) {
        setDateRange(response.data);
        setRetrainStart(response.data.min_date);
        setRetrainEnd(response.data.max_date);
      }
    } catch (error) { console.error('Failed to fetch date range:', error); }
  }, []);

  useEffect(() => {
    if (apiStatus?.graph_loaded) {
      fetchGraphData();
      fetchBacktestData();
      fetchDateRange();
    }
  }, [apiStatus, fetchGraphData, fetchBacktestData, fetchDateRange]);

  useEffect(() => {
    if (analysisRun) {
      fetchPredictions();
      fetchSimulations();
      fetchMitigations();
      fetchStories();
    }
  }, [analysisRun, fetchPredictions, fetchSimulations, fetchMitigations, fetchStories]);

  const handleStartPipeline = async () => {
    setPipelineRunning(true);
    setPipelineError(null);
    setPipelineSummary(null);
    setPipelineSteps(PIPELINE_STEPS.map(s => ({ ...s, status: 'pending', timestamp: null })));

    let backendResult = null;
    let backendDone = false;
    const apiCall = axios.post(`${API_BASE_URL}/run-analysis`)
      .then(res => { backendResult = res.data; backendDone = true; })
      .catch(() => { backendResult = { error: 'Failed to connect to backend' }; backendDone = true; });

    for (let i = 0; i < PIPELINE_STEPS.length; i++) {
      setPipelineSteps(prev => prev.map((s, idx) =>
        idx === i ? { ...s, status: 'running', timestamp: new Date().toLocaleTimeString() } : s
      ));
      if (i === PIPELINE_STEPS.length - 1) {
        if (!backendDone) await apiCall;
      } else {
        const delay = backendDone ? 250 : 1200;
        await new Promise(r => setTimeout(r, delay));
      }
      setPipelineSteps(prev => prev.map((s, idx) =>
        idx === i ? { ...s, status: 'done', timestamp: new Date().toLocaleTimeString() } : s
      ));
    }

    if (backendResult?.error) {
      setPipelineError(backendResult.error);
    } else {
      setPipelineSummary(backendResult);
      setAnalysisRun(true);
      setPipelineCompletedAt(new Date().toLocaleTimeString());
      setLastUpdate(0);
    }
    setPipelineRunning(false);
  };

  const handleRetrain = async () => {
    setRetrainLoading(true);
    setRetrainMeta(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/retrain-model`, {
        start_date: retrainStart,
        end_date: retrainEnd,
      });
      if (response.data.error) {
        setRetrainMeta({ error: response.data.error });
      } else {
        setRetrainMeta(response.data);
        setAnalysisRun(true);
        setLastUpdate(0);
        fetchPredictions();
        fetchSimulations();
        fetchMitigations();
        fetchStories();
        fetchBacktestData();
      }
    } catch (error) { setRetrainMeta({ error: 'Failed to connect to backend' }); }
    setRetrainLoading(false);
  };

  const top10 = predictions ? [...predictions].sort((a, b) => b.risk - a.risk).slice(0, 10) : [];
  const bottom10 = predictions ? [...predictions].sort((a, b) => a.risk - b.risk).slice(0, 10) : [];
  const topSimulation = simulations ? [...simulations].sort((a, b) => (b.worst_case - b.best_case) - (a.worst_case - a.best_case))[0] : null;
  const singleSimData = topSimulation ? [
    { name: 'Best Case', delay: +topSimulation.best_case.toFixed(1), fill: '#00ff88' },
    { name: 'Expected', delay: +topSimulation.expected_delay_hours.toFixed(1), fill: '#3b82f6' },
    { name: 'Worst Case', delay: +topSimulation.worst_case.toFixed(1), fill: '#ff3864' },
  ] : [];

  const mitigationList = mitigations
    ? Object.entries(mitigations)
        .map(([uid, m]) => ({ trip_uuid: uid, ...m }))
        .filter(m => m.solana_tx)
        .sort((a, b) => (b.original_risk || 0) - (a.original_risk || 0))
        .slice(0, 15)
    : [];

  const onChainCount = mitigationList.length;
  const costOfInaction = predictions ? predictions.reduce((sum, p) => sum + p.risk * p.expected_delay_hours * CARGO_COST_PER_HOUR, 0) : 0;
  const isHealthy = apiStatus?.status === 'healthy';

  const renderNotRunPlaceholder = (tabName) => (
    <div className="panel not-run-panel">
      <div className="not-run-content">
        <div className="not-run-icon">&#9888;</div>
        <h3>Analysis Not Yet Executed</h3>
        <p>{tabName} data will be available after running the pipeline.</p>
        <button className="run-btn small" onClick={() => setActiveTab('overview')}>Go to Overview</button>
      </div>
    </div>
  );

  const renderPipelineLog = () => (
    <div className="pipeline-log">
      <div className="pipeline-log-title">Pipeline Execution Log</div>
      {pipelineSteps.map((step) => (
        <div key={step.key} className={`pipeline-step ${step.status}`}>
          <div className="step-icon">
            {step.status === 'pending' && <span className="step-dot" />}
            {step.status === 'running' && <span className="step-spinner" />}
            {step.status === 'done' && <span className="step-check">{'\u2713'}</span>}
          </div>
          <span className="step-label">{step.label}</span>
          {step.timestamp && <span className="step-timestamp">{step.timestamp}</span>}
        </div>
      ))}
      {pipelineError && <div className="pipeline-error">Error: {pipelineError}</div>}
      {pipelineSummary && (
        <div className="pipeline-summary">
          Pipeline complete — {pipelineSummary.total_trips} trips analyzed, {pipelineSummary.high_risk_trips} high-risk.
        </div>
      )}
    </div>
  );

  return (
    <div>
      <div className="grain" />
      <header className="app-header">
        <div className="header-content">
          <h1>CHAIN-REACTION</h1>
          <div className="status-bar">
            <div className="status-item">
              <span className={`status-dot ${isHealthy ? 'live' : 'error'}`} />
              <span>{loading ? 'Connecting...' : isHealthy ? 'System Online' : 'Offline'}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="tabs">
        {[{ key: 'overview', label: 'Overview' }, { key: 'analysis', label: 'Analysis' }, { key: 'mitigations', label: 'Mitigations' }].map(tab => (
          <button
            key={tab.key}
            className={`tab ${activeTab === tab.key ? 'active' : ''} ${!analysisRun && tab.key !== 'overview' ? 'tab-disabled' : ''}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <main className="app-main">
        <div className={`tab-content ${activeTab === 'overview' ? 'active' : ''}`}>
          <div className="landing-hero">
            <div className="hero-text">
              <h2 className="hero-title">Predictive Digital Twin for Global Supply Chains</h2>
              <p className="hero-description">
                Chain-Reaction forecasts disruption risk using LightGBM ML models, runs Monte Carlo simulations, and triggers Solana-recorded mitigations.
              </p>
            </div>
            <div className="hero-stats">
              <div className="hero-stat">
                <span className="hero-stat-value blue">{graphData ? graphData.nodes.length : '--'}</span>
                <span className="hero-stat-label">Nodes</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-value safe">{analysisRun ? 'Done' : 'Ready'}</span>
                <span className="hero-stat-label">Pipeline</span>
              </div>
            </div>
          </div>

          {backtestData && (
            <div className="panel" style={{ marginBottom: '1.5rem' }}>
              <div className="panel-header"><span className="panel-title">Model Validation</span></div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={backtestData.points}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.15)" />
                  <XAxis dataKey="ts" tick={{ fill: '#c0c7d6', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#c0c7d6', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: 'rgba(20, 25, 32, 0.95)', border: '1px solid rgba(139,146,168,0.25)' }} />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="predicted" stroke="#00ff88" strokeWidth={2} dot={false} strokeDasharray="6 3" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {dateRange && (
            <div className="panel retrain-panel" style={{ marginBottom: '1.5rem' }}>
              <div className="panel-header"><span className="panel-title">Model Training Window</span></div>
              <div className="retrain-controls">
                <input type="date" className="date-input" value={retrainStart} onChange={e => setRetrainStart(e.target.value)} disabled={retrainLoading} />
                <input type="date" className="date-input" value={retrainEnd} onChange={e => setRetrainEnd(e.target.value)} disabled={retrainLoading} />
                <button className={`retrain-btn ${retrainLoading ? 'loading' : ''}`} onClick={handleRetrain} disabled={retrainLoading}>
                  {retrainLoading ? 'Retraining...' : 'Retrain Model'}
                </button>
              </div>
            </div>
          )}

          <div className="run-section">
            <button className={`run-btn ${pipelineRunning ? 'loading' : ''} ${analysisRun ? 'completed' : ''}`} onClick={handleStartPipeline} disabled={pipelineRunning || analysisRun || !isHealthy}>
              {pipelineRunning ? 'Running Pipeline...' : analysisRun ? 'Pipeline Complete' : 'Start Pipeline'}
            </button>
          </div>
          {pipelineSteps.length > 0 && renderPipelineLog()}
        </div>

        <div className={`tab-content ${activeTab === 'analysis' ? 'active' : ''}`}>
          {!analysisRun ? renderNotRunPlaceholder('Analysis') : (
            <>
              <div className="cost-metric">
                <span className="cost-label">Estimated Cost of Inaction</span>
                <span className="cost-value">${costOfInaction.toLocaleString()}</span>
              </div>
              <div className="grid">
                <div className="panel">
                  <div className="panel-header"><span className="panel-title">Top Riskiest</span></div>
                  <table className="pred-table">
                    <thead><tr><th>Trip ID</th><th>Risk %</th></tr></thead>
                    <tbody>{top10.map(p => (<tr key={p.trip_uuid}><td className="mono">{readableId(p.trip_uuid)}</td><td>{(p.risk * 100).toFixed(1)}%</td></tr>))}</tbody>
                  </table>
                </div>
              </div>
              {topSimulation && (
                <div className="panel">
                  <div className="panel-header"><span className="panel-title">Monte Carlo Volatility</span></div>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={singleSimData}><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="delay">{singleSimData.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar></BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </div>

        <div className={`tab-content ${activeTab === 'mitigations' ? 'active' : ''}`}>
          {!analysisRun ? renderNotRunPlaceholder('Mitigations') : (
            <div className="panel">
              <div className="panel-header"><span className="panel-title">On-Chain Mitigations ({onChainCount})</span></div>
              {mitigationList.map(m => (
                <div key={m.trip_uuid} className="mitigation-card">
                  <div className="mitigation-header">
                    <span className="mitigation-trip-id">{readableId(m.trip_uuid)}</span>
                    <a href={`${SOLANA_EXPLORER}/${m.solana_tx}?cluster=devnet`} target="_blank" rel="noopener noreferrer" className="solana-link">View TX</a>
                  </div>
                  <div className="mitigation-action">{m.strategy}</div>
                  <div className="decision-hash"><code>{m.solana_tx}</code></div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;