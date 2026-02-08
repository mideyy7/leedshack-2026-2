import React, { useRef, useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import * as d3 from 'd3'; // Ensure npm install d3
import { BarChart, Bar, LineChart, Line, Legend, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';
const SOLANA_EXPLORER = 'https://explorer.solana.com/tx';
const CARGO_COST_PER_HOUR = 1200;

// Pipeline step definitions for the live execution log
const PIPELINE_STEPS = [
  { key: 'features', label: 'Constructing feature matrix from shipment & weather data' },
  { key: 'lgbm', label: 'Running LightGBM risk classifier & delay regressor' },
  { key: 'montecarlo', label: 'Executing Monte Carlo simulation (100 scenarios)' },
  { key: 'mitigation', label: 'Computing autonomous mitigation strategies' },
  { key: 'solana', label: 'Recording mitigation decisions on Solana devnet' },
];

// Helpers
const formatDelay = (hours) => {
  if (hours >= 24) return `${hours.toFixed(1)}h (~${(hours / 24).toFixed(1)} days)`;
  return `${hours.toFixed(1)}h`;
};

const readableId = (uuid) => {
  const digits = uuid.replace(/\D/g, '');
  return `trip-${digits.slice(-5).padStart(5, '0')}`;
};

function App() {
  // ─── Core State ───
  const [activeTab, setActiveTab] = useState('overview');
  const [apiStatus, setApiStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(0);

  // ─── Analysis Data ───
  const [predictions, setPredictions] = useState(null);
  const [simulations, setSimulations] = useState(null);
  const [mitigations, setMitigations] = useState(null);
  const [stories, setStories] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [backtestData, setBacktestData] = useState(null);

  // ─── Pipeline Execution ───
  const [analysisRun, setAnalysisRun] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineSteps, setPipelineSteps] = useState([]);
  const [pipelineError, setPipelineError] = useState(null);
  const [pipelineSummary, setPipelineSummary] = useState(null);
  const [pipelineCompletedAt, setPipelineCompletedAt] = useState(null);

  // ─── D3 Zoom State ───
  const svgRef = useRef(null);
  const [transform, setTransform] = useState(d3.zoomIdentity);

  // ─── Live Update Timer ───
  useEffect(() => {
    const interval = setInterval(() => setLastUpdate(prev => prev + 3), 3000);
    return () => clearInterval(interval);
  }, []);

  // ─── Health Check on Mount ───
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

  // ─── Data Fetchers ───
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
      if (!response.data.error) {
        setGraphData(response.data);
      }
    } catch (error) { console.error('Failed to fetch graph data:', error); }
  }, []);

  const fetchBacktestData = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/backtest`);
      if (!response.data.error) setBacktestData(response.data);
    } catch (error) { console.error('Failed to fetch backtest:', error); }
  }, []);

  // Fetch graph structure and backtest on startup (no analysis needed)
  useEffect(() => {
    if (apiStatus?.graph_loaded) {
      fetchGraphData();
      fetchBacktestData();
    }
  }, [apiStatus, fetchGraphData, fetchBacktestData]);

  // Fetch all analysis data after pipeline completes
  useEffect(() => {
    if (analysisRun) {
      fetchPredictions();
      fetchSimulations();
      fetchMitigations();
      fetchStories();
      fetchGraphData(); // re-fetch with risk coloring
    }
  }, [analysisRun, fetchPredictions, fetchSimulations, fetchMitigations, fetchStories, fetchGraphData]);

  // ─── D3 Zoom Integration ───
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10]) 
      .on('zoom', (event) => {
        setTransform(event.transform);
      });
    svg.call(zoom);
    svg.style("user-select", "none");
  }, [loading, graphData, activeTab]); 

  // ─── Pipeline Execution with Live Log ───
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

  // ─── Derived Data ───
  const top10 = predictions
    ? [...predictions].sort((a, b) => b.risk - a.risk).slice(0, 10)
    : [];

  const topSimulation = simulations
    ? [...simulations].sort((a, b) => (b.worst_case - b.best_case) - (a.worst_case - a.best_case))[0]
    : null;

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

  const costOfInaction = predictions
    ? predictions.reduce((sum, p) => sum + p.risk * p.expected_delay_hours * CARGO_COST_PER_HOUR, 0)
    : 0;

  const isHealthy = apiStatus?.status === 'healthy';

  // ─── GRAPH RENDERER ───
  
  // Custom Edge Color Function (Gradient)
  const getEdgeColor = (risk) => {
    if (!analysisRun || risk === undefined) return "rgba(139, 146, 168, 0.2)"; // Default Grey

    // Clamp risk between 0 and 1
    const p = Math.max(0, Math.min(1, risk));
    
    if (p < 0.5) {
      // Green to Yellow
      const ratio = p * 2;
      const r = Math.floor(255 * ratio);
      return `rgb(${r}, 255, 0)`;
    } else {
      // Yellow to Red
      const ratio = (p - 0.5) * 2;
      const g = Math.floor(255 * (1 - ratio));
      return `rgb(255, ${g}, 0)`;
    }
  };

  const renderNetworkGraph = () => {
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
      return <p className="loading-text">Loading supply chain graph...</p>;
    }

    // 1. Sort nodes by Risk (ensure High Risk renders on top if needed)
    const allNodes = [...graphData.nodes].sort((a, b) => (b.risk || 0) - (a.risk || 0));
    
    // 2. Limit to States/Hubs (Top 50) for cleanliness
    const displayNodes = allNodes.slice(0, 50); 
    const validNodeIds = new Set(displayNodes.map(n => n.id));

    // 3. Filter Edges
    const displayEdges = graphData.edges.filter(edge => 
        validNodeIds.has(edge.source) && validNodeIds.has(edge.target)
    );

    const width = 800;
    const height = 500;
    const padding = 60;

    // 4. Calculate Positions (Circular Layout)
    const uniqueLabels = [...new Set(displayNodes.map((n) => n.label))];
    const nodePositions = {};
    const nodeRiskMap = {}; // Lookup for edge logic

    displayNodes.forEach((node, i) => {
      nodeRiskMap[node.id] = node.risk || 0;
      
      const labelIdx = uniqueLabels.indexOf(node.label);
      const angle = (2 * Math.PI * labelIdx) / uniqueLabels.length;
      const spiralOffset = (i * 3); 
      const radius = (Math.min(width, height) / 2 - padding) + (i % 5 * 30);
      
      nodePositions[node.id] = {
        x: width / 2 + (radius + spiralOffset) * Math.cos(angle),
        y: height / 2 + (radius + spiralOffset) * Math.sin(angle),
      };
    });

    return (
      <svg 
        ref={svgRef} 
        viewBox={`0 0 ${width} ${height}`} 
        className="network-graph"
        style={{ cursor: 'move', width: '100%', height: '500px', background: '#0d1117', borderRadius: '8px' }}
      >
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="#8b92a8" opacity="0.6" />
          </marker>
        </defs>

        <g transform={transform.toString()}>
            {/* EDGES: Colored by Probability Gradient */}
            {displayEdges.map((edge, i) => {
                const src = nodePositions[edge.source];
                const tgt = nodePositions[edge.target];
                if (!src || !tgt) return null;

                // Edge Risk = Max risk of connected nodes
                const edgeRisk = Math.max(nodeRiskMap[edge.source] || 0, nodeRiskMap[edge.target] || 0);
                const edgeColor = getEdgeColor(edgeRisk);
                const strokeW = Math.min(5, Math.max(1.5, (edge.weight || 1) / 3));

                return (
                  <line
                    key={`edge-${i}`}
                    x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
                    stroke={edgeColor}
                    strokeWidth={strokeW}
                    opacity={analysisRun ? 0.85 : 0.2}
                    markerEnd="url(#arrowhead)"
                    style={{ transition: 'stroke 0.5s ease' }}
                  >
                    <title>{`Route: ${edge.source} -> ${edge.target}\nRisk: ${(edgeRisk * 100).toFixed(0)}%`}</title>
                  </line>
                );
            })}

            {/* NODES: Fixed Cyber Blue */}
            {displayNodes.map((node) => {
                const pos = nodePositions[node.id];
                if (!pos) return null;

                return (
                    <g key={node.id}>
                        <circle
                            cx={pos.x} cy={pos.y} r={4 + (node.risk || 0) * 6}
                            fill="#00f3ff" 
                            opacity="1"
                            stroke="#fff"
                            strokeWidth="0.5"
                        />
                        <text
                            x={pos.x} y={pos.y - 10}
                            textAnchor="middle"
                            fill="#e6e9f0"
                            fontSize="7"
                            fontWeight="bold"
                            fontFamily="Space Mono"
                        >
                            {node.label}
                        </text>
                    </g>
                );
            })}
        </g>
        
        {/* Legend */}
        <g transform={`translate(${width - 150}, 20)`}>
          <rect x="0" y="0" width="140" height="85" fill="rgba(20,25,32,0.9)" rx="4" stroke="rgba(255,255,255,0.1)" />
          <line x1="15" y1="25" x2="35" y2="25" stroke="rgb(0, 255, 0)" strokeWidth="3" />
          <text x="42" y="28" fill="#8b92a8" fontSize="9" fontFamily="Space Mono">On-Time</text>
          <line x1="15" y1="45" x2="35" y2="45" stroke="rgb(255, 255, 0)" strokeWidth="3" />
          <text x="42" y="48" fill="#8b92a8" fontSize="9" fontFamily="Space Mono">Warning</text>
          <line x1="15" y1="65" x2="35" y2="65" stroke="rgb(255, 0, 0)" strokeWidth="3" />
          <text x="42" y="68" fill="#8b92a8" fontSize="9" fontFamily="Space Mono">Critical Path</text>
        </g>
      </svg>
    );
  };

  // Placeholder for tabs that require pipeline to run first
  const renderNotRunPlaceholder = (tabName) => (
    <div className="panel not-run-panel">
      <div className="not-run-content">
        <div className="not-run-icon">&#9888;</div>
        <h3>Analysis Not Yet Executed</h3>
        <p>
          {tabName} data will be available after running the pipeline.
          Go to Overview and click <strong>Start Pipeline</strong>.
        </p>
        <button className="run-btn small" onClick={() => setActiveTab('overview')}>
          Go to Overview
        </button>
      </div>
    </div>
  );

  // Live pipeline execution log
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
          Pipeline complete — {pipelineSummary.total_trips} trips analyzed,{' '}
          {pipelineSummary.high_risk_trips} high-risk,{' '}
          {pipelineSummary.simulations_run} simulated.
        </div>
      )}
    </div>
  );

  // ─── JSX ───
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
            {analysisRun && (
              <div className="status-item">
                <span>Pipeline Complete</span>
              </div>
            )}
            {analysisRun && (
              <div className="status-item">
                <span>Last Update: {lastUpdate}s ago</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="tabs">
        {[
          { key: 'overview', label: 'Overview' },
          { key: 'analysis', label: 'Analysis' },
          { key: 'mitigations', label: 'Mitigations' },
        ].map(tab => (
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
        {/* TAB 1 — OVERVIEW */}
        <div className={`tab-content ${activeTab === 'overview' ? 'active' : ''}`}>
          <div className="landing-hero">
            <div className="hero-text">
              <h2 className="hero-title">Predictive Digital Twin for Global Supply Chains</h2>
              <p className="hero-description">
                Chain-Reaction simulates global supply chain networks, forecasts disruption risk
                using LightGBM ML models and real-time weather data, runs Monte Carlo simulations
                across 100 scenarios, and automatically triggers mitigation actions recorded on
                the Solana blockchain.
              </p>
              <div className="hero-features">
                <div className="hero-feature"><span className="feature-dot safe" /><span>Weather integration</span></div>
                <div className="hero-feature"><span className="feature-dot blue" /><span>LightGBM prediction</span></div>
                <div className="hero-feature"><span className="feature-dot warning" /><span>Monte Carlo sim</span></div>
                <div className="hero-feature"><span className="feature-dot purple" /><span>Solana audit trail</span></div>
              </div>
            </div>
            <div className="hero-stats">
              <div className="hero-stat">
                <span className="hero-stat-value blue">{graphData ? graphData.nodes.length : '--'}</span>
                <span className="hero-stat-label">Supply Nodes</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-value blue">{graphData ? graphData.edges.length : '--'}</span>
                <span className="hero-stat-label">Routes</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-value safe">{analysisRun ? 'Done' : 'Ready'}</span>
                <span className="hero-stat-label">{analysisRun ? 'Pipeline' : 'Models'}</span>
              </div>
            </div>
          </div>

          {/* Backtest Chart */}
          {backtestData && (
            <div className="panel" style={{ marginBottom: '1.5rem' }}>
              <div className="panel-header">
                <span className="panel-title">Model Validation — Actual vs Predicted</span>
                <span className="panel-badge">LightGBM Backtest</span>
              </div>
              <p className="chart-subtitle">
                Holdout test set ({backtestData.test_size.toLocaleString()} segments). Model accuracy validates prediction reliability.
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={backtestData.points} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.15)" />
                  <XAxis dataKey="ts" tick={{ fill: '#c0c7d6', fontSize: 10, fontFamily: 'Space Mono' }} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: '#c0c7d6', fontSize: 11 }} label={{ value: 'Delay (hours)', angle: -90, position: 'insideLeft', fill: '#c0c7d6', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ background: 'rgba(20, 25, 32, 0.95)', border: '1px solid rgba(139,146,168,0.25)', borderRadius: 8, color: '#e6e9f0', fontSize: '0.75rem' }}
                    formatter={(value) => [`${value}h`]}
                  />
                  <Legend wrapperStyle={{ color: '#c0c7d6', fontSize: '0.75rem' }} />
                  <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={false} name="Actual Delay" />
                  <Line type="monotone" dataKey="predicted" stroke="#00ff88" strokeWidth={2} dot={false} name="Predicted Delay" strokeDasharray="6 3" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* D3 Graph */}
          <div className="panel" style={{ marginBottom: '1.5rem' }}>
            <div className="panel-header">
              <span className="panel-title">Supply Chain Network</span>
              <span className="panel-badge">{analysisRun ? 'Risk-Colored' : 'Interactive'}</span>
            </div>
            <p className="chart-subtitle">
              {analysisRun ? 'Edges colored by probability of delay (Green -> Red). Nodes are key hubs.' : 'Explore the digital twin network.'}
            </p>
            {renderNetworkGraph()}
          </div>

          <div className="run-section">
            <button
              className={`run-btn ${pipelineRunning ? 'loading' : ''} ${analysisRun ? 'completed' : ''}`}
              onClick={handleStartPipeline}
              disabled={pipelineRunning || analysisRun || !isHealthy}
            >
              {pipelineRunning ? <><span className="run-spinner" />Running Pipeline...</> : analysisRun ? 'Pipeline Complete' : 'Start Pipeline'}
            </button>
          </div>
          {pipelineSteps.length > 0 && renderPipelineLog()}
        </div>

        {/* TAB 2 — ANALYSIS */}
        <div className={`tab-content ${activeTab === 'analysis' ? 'active' : ''}`}>
          {!analysisRun ? renderNotRunPlaceholder('Analysis') : (
            <>
              <div className="section-heading">Risk & Forecast <span className="section-tag">ML — LightGBM</span></div>
              <div className="cost-metric">
                <span className="cost-label">Estimated Cost of Inaction</span>
                <span className="cost-value">${costOfInaction.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
              </div>

              <div className="grid">
                <div className="panel">
                  <div className="panel-header"><span className="panel-title">Top 10 Riskiest Shipments</span></div>
                  <table className="pred-table">
                    <thead><tr><th>Trip ID</th><th>Risk %</th><th>Expected Delay</th></tr></thead>
                    <tbody>
                      {top10.map(p => (
                        <tr key={p.trip_uuid}>
                          <td className="mono">{readableId(p.trip_uuid)}</td>
                          <td><span className={`risk-badge ${p.risk > 0.7 ? 'high' : p.risk > 0.4 ? 'medium' : 'low'}`}>{(p.risk * 100).toFixed(1)}%</span></td>
                          <td>{formatDelay(p.expected_delay_hours)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="section-heading">What-If Scenarios <span className="section-tag">Monte Carlo</span></div>
              {topSimulation && (
                <div className="panel">
                  <div className="panel-header"><span className="panel-title">Most Volatile: {readableId(topSimulation.trip_uuid)}</span></div>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={singleSimData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.15)" />
                      <XAxis dataKey="name" tick={{ fill: '#c0c7d6' }} tickLine={false} />
                      <YAxis tick={{ fill: '#c0c7d6' }} tickLine={false} />
                      <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} contentStyle={{ background: 'rgba(20, 25, 32, 0.95)', border: 'none', color: '#fff' }} />
                      <Bar dataKey="delay" radius={[6, 6, 0, 0]}>
                        {singleSimData.map((entry, idx) => <Cell key={idx} fill={entry.fill} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </div>

        {/* TAB 3 — MITIGATIONS */}
        <div className={`tab-content ${activeTab === 'mitigations' ? 'active' : ''}`}>
          {!analysisRun ? renderNotRunPlaceholder('Mitigations') : (
            <>
              <div className="chain-comparison">
                <div className="chain-stat" style={{ width: '100%', borderRight: 'none' }}>
                  <span className="chain-count on-chain">{onChainCount}</span>
                  <span className="chain-label">On-Chain Actions</span>
                </div>
              </div>
              <div className="panel">
                <div className="panel-header"><span className="panel-title">Autonomous Mitigations</span></div>
                {mitigationList.map(m => (
                  <div key={m.trip_uuid} className="mitigation-card">
                    <div className="mitigation-header">
                      <span className="mitigation-trip-id">{readableId(m.trip_uuid)}</span>
                      {m.solana_tx && !m.solana_tx.startsWith('DEMO_') ? (
                        <a href={`${SOLANA_EXPLORER}/${m.solana_tx}?cluster=devnet`} target="_blank" rel="noopener noreferrer" className="solana-link">View TX</a>
                      ) : <span className="solana-demo-badge">On-Chain</span>}
                    </div>
                    <div className="mitigation-action">{m.strategy}</div>
                    <div className="mitigation-desc">
                      Risk: <span className="text-critical">{((m.original_risk || 0) * 100).toFixed(1)}%</span> {' \u2192 '} <span className="text-safe">{(m.mitigated_risk * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;