import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
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

  // ─── Interactive Graph ───
  const [nodePositions, setNodePositions] = useState({});
  const [dragNode, setDragNode] = useState(null);
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: 800, h: 500 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState(null);
  const svgRef = useRef(null);

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
        // Initialize node positions with circular layout on first load
        if (response.data.nodes && Object.keys(nodePositions).length === 0) {
          const nodes = response.data.nodes;
          const uniqueLabels = [...new Set(nodes.map(n => n.label))];
          const w = 800, h = 500, pad = 80;
          const positions = {};
          nodes.forEach((node, i) => {
            const idx = uniqueLabels.indexOf(node.label);
            const angle = (2 * Math.PI * idx) / uniqueLabels.length;
            const r = Math.min(w, h) / 2 - pad;
            const jitter = (i % 3) * 18;
            positions[node.id] = {
              x: w / 2 + (r - jitter) * Math.cos(angle),
              y: h / 2 + (r - jitter) * Math.sin(angle),
            };
          });
          setNodePositions(positions);
        }
      }
    } catch (error) { console.error('Failed to fetch graph data:', error); }
  }, [nodePositions]);

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

  // ─── Pipeline Execution with Live Log ───
  const handleStartPipeline = async () => {
    setPipelineRunning(true);
    setPipelineError(null);
    setPipelineSummary(null);
    setPipelineSteps(PIPELINE_STEPS.map(s => ({ ...s, status: 'pending', timestamp: null })));

    // Fire the backend call (runs in parallel with animation)
    let backendResult = null;
    let backendDone = false;
    const apiCall = axios.post(`${API_BASE_URL}/run-analysis`)
      .then(res => { backendResult = res.data; backendDone = true; })
      .catch(() => { backendResult = { error: 'Failed to connect to backend' }; backendDone = true; });

    // Animate pipeline steps with live timestamps
    for (let i = 0; i < PIPELINE_STEPS.length; i++) {
      // Mark current step as running
      setPipelineSteps(prev => prev.map((s, idx) =>
        idx === i ? { ...s, status: 'running', timestamp: new Date().toLocaleTimeString() } : s
      ));

      // On the last step, wait for the backend to actually finish
      if (i === PIPELINE_STEPS.length - 1) {
        if (!backendDone) await apiCall;
      } else {
        // Speed up remaining steps once backend responds
        const delay = backendDone ? 250 : 1200;
        await new Promise(r => setTimeout(r, delay));
      }

      // Mark current step as done
      setPipelineSteps(prev => prev.map((s, idx) =>
        idx === i ? { ...s, status: 'done', timestamp: new Date().toLocaleTimeString() } : s
      ));
    }

    // Handle final result
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

  // ─── Interactive Graph: Drag & Pan Handlers ───
  const getSVGPoint = (e) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  };

  const handleNodeMouseDown = (e, nodeId) => {
    e.stopPropagation();
    setDragNode(nodeId);
  };

  const handleSvgMouseDown = (e) => {
    if (dragNode) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX, y: e.clientY, vx: viewBox.x, vy: viewBox.y });
  };

  const handleSvgMouseMove = (e) => {
    if (dragNode) {
      const pt = getSVGPoint(e);
      setNodePositions(prev => ({ ...prev, [dragNode]: { x: pt.x, y: pt.y } }));
    } else if (isPanning && panStart) {
      const svg = svgRef.current;
      if (!svg) return;
      const scale = viewBox.w / svg.clientWidth;
      setViewBox(prev => ({
        ...prev,
        x: panStart.vx - (e.clientX - panStart.x) * scale,
        y: panStart.vy - (e.clientY - panStart.y) * scale,
      }));
    }
  };

  const handleSvgMouseUp = () => {
    setDragNode(null);
    setIsPanning(false);
    setPanStart(null);
  };

  // ─── Derived Data ───

  // Top 10 riskiest shipments for table (Descending risk)
  const top10 = predictions
    ? [...predictions].sort((a, b) => b.risk - a.risk).slice(0, 10)
    : [];

  // Bottom 10 riskiest (Safest) shipments for table (Ascending risk)
  const bottom10 = predictions
    ? [...predictions].sort((a, b) => a.risk - b.risk).slice(0, 10)
    : [];

  // Most volatile shipment for What-If
  const topSimulation = simulations
    ? [...simulations].sort((a, b) => (b.worst_case - b.best_case) - (a.worst_case - a.best_case))[0]
    : null;

  // Single-shipment bar chart data
  const singleSimData = topSimulation ? [
    { name: 'Best Case', delay: +topSimulation.best_case.toFixed(1), fill: '#00ff88' },
    { name: 'Expected', delay: +topSimulation.expected_delay_hours.toFixed(1), fill: '#3b82f6' },
    { name: 'Worst Case', delay: +topSimulation.worst_case.toFixed(1), fill: '#ff3864' },
  ] : [];

  // FILTERED: Only show mitigations that have a solana_tx (On-Chain Only)
  const mitigationList = mitigations
    ? Object.entries(mitigations)
        .map(([uid, m]) => ({ trip_uuid: uid, ...m }))
        .filter(m => m.solana_tx) // Exclude off-chain
        .sort((a, b) => (b.original_risk || 0) - (a.original_risk || 0)) // Sort by risk
        .slice(0, 15)
    : [];

  // Count of On-Chain (since we filtered, this is just the list length)
  const onChainCount = mitigationList.length;

  // Estimated cost of inaction across all predicted shipments
  const costOfInaction = predictions
    ? predictions.reduce((sum, p) => sum + p.risk * p.expected_delay_hours * CARGO_COST_PER_HOUR, 0)
    : 0;

  const isHealthy = apiStatus?.status === 'healthy';

  // ─── Renderers ───

  // Interactive SVG network graph
  const renderInteractiveGraph = () => {
    if (!graphData?.nodes?.length) {
      return <p className="loading-text">Loading supply chain graph...</p>;
    }

    const { nodes, edges } = graphData;

    const getNodeRadius = (node) => {
      if (analysisRun) return 6 + node.risk * 12;
      return { hub: 10, port: 9, warehouse: 8, supplier: 7 }[node.type] || 7;
    };

    return (
      <svg
        ref={svgRef}
        viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
        className="network-graph interactive"
        onMouseDown={handleSvgMouseDown}
        onMouseMove={handleSvgMouseMove}
        onMouseUp={handleSvgMouseUp}
        onMouseLeave={handleSvgMouseUp}
      >
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="#8b92a8" opacity="0.5" />
          </marker>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Edges */}
        {edges.slice(0, 50).map((edge, i) => {
          const src = nodePositions[edge.source];
          const tgt = nodePositions[edge.target];
          if (!src || !tgt) return null;
          return (
            <line key={`e-${i}`}
              x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
              stroke={edge.color} strokeWidth="1" opacity="0.35"
              markerEnd="url(#arrowhead)"
            />
          );
        })}

        {/* Nodes */}
        {nodes.slice(0, 25).map(node => {
          const pos = nodePositions[node.id];
          if (!pos) return null;
          const r = getNodeRadius(node);
          return (
            <g key={node.id} style={{ cursor: 'grab' }}
              onMouseDown={e => handleNodeMouseDown(e, node.id)}>
              <circle cx={pos.x} cy={pos.y} r={r}
                fill={node.color} opacity="0.85"
                stroke={node.color} strokeWidth="2"
                filter={r > 9 ? 'url(#glow)' : undefined}
              />
              <text x={pos.x} y={pos.y - r - 4}
                textAnchor="middle" fill="#8b92a8"
                fontSize="7" fontFamily="Space Mono">
                {node.label}
              </text>
              {!analysisRun && (
                <circle cx={pos.x + r - 2} cy={pos.y - r + 2} r="2.5"
                  fill="#00ff88" stroke="none" />
              )}
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${viewBox.x + viewBox.w - 140}, ${viewBox.y + 15})`}>
          <rect x="0" y="0" width="130" height={analysisRun ? 75 : 55}
            fill="rgba(20,25,32,0.85)" rx="4" stroke="rgba(139,146,168,0.12)" />
          {analysisRun ? (
            <>
              <circle cx="15" cy="18" r="5" fill="#00ff88" />
              <text x="28" y="22" fill="#8b92a8" fontSize="8" fontFamily="Space Mono">Low Risk</text>
              <circle cx="15" cy="38" r="5" fill="#ffd700" />
              <text x="28" y="42" fill="#8b92a8" fontSize="8" fontFamily="Space Mono">Medium Risk</text>
              <circle cx="15" cy="58" r="5" fill="#ff3864" />
              <text x="28" y="62" fill="#8b92a8" fontSize="8" fontFamily="Space Mono">High Risk</text>
            </>
          ) : (
            <>
              <circle cx="15" cy="18" r="5" fill="#3b82f6" />
              <text x="28" y="22" fill="#8b92a8" fontSize="8" fontFamily="Space Mono">Supply Node</text>
              <circle cx="15" cy="38" r="2.5" fill="#00ff88" />
              <text x="28" y="42" fill="#8b92a8" fontSize="8" fontFamily="Space Mono">On-Time</text>
            </>
          )}
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
      {pipelineError && (
        <div className="pipeline-error">Error: {pipelineError}</div>
      )}
      {pipelineSummary && (
        <div className="pipeline-summary">
          Pipeline complete — {pipelineSummary.total_trips} trips analyzed,{' '}
          {pipelineSummary.high_risk_trips} high-risk,{' '}
          {pipelineSummary.simulations_run} simulated,{' '}
          {pipelineSummary.mitigations_computed} mitigated,{' '}
          {pipelineSummary.stories_generated} stories generated.
        </div>
      )}
    </div>
  );

  // ─── JSX ───
  return (
    <div>
      <div className="grain" />

      {/* ═══ Header ═══ */}
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

      {/* ═══ Tabs ═══ */}
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

        {/* ═══════════════════════════════════════════════
            TAB 1 — OVERVIEW
            Always shows: description, graph, pipeline button/log
        ═══════════════════════════════════════════════ */}
        <div className={`tab-content ${activeTab === 'overview' ? 'active' : ''}`}>

          {/* Brief project description */}
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

          {/* Model Validation: Actual vs Predicted (backtest chart) */}
          {backtestData && (
            <div className="panel" style={{ marginBottom: '1.5rem' }}>
              <div className="panel-header">
                <span className="panel-title">Model Validation — Actual vs Predicted</span>
                <span className="panel-badge">LightGBM Backtest</span>
              </div>
              <p className="chart-subtitle">
                Holdout test set ({backtestData.test_size.toLocaleString()} segments,{' '}
                trained on {backtestData.train_size.toLocaleString()}). Model accuracy validates prediction reliability.
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={backtestData.points} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.15)" />
                  <XAxis
                    dataKey="ts"
                    tick={{ fill: '#c0c7d6', fontSize: 10, fontFamily: 'Space Mono' }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fill: '#c0c7d6', fontSize: 11 }}
                    label={{ value: 'Delay (hours)', angle: -90, position: 'insideLeft', fill: '#c0c7d6', fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: 'rgba(20, 25, 32, 0.95)',
                      border: '1px solid rgba(139,146,168,0.25)',
                      borderRadius: 8,
                      color: '#e6e9f0',
                      padding: '0.75rem 1rem',
                      boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                      fontFamily: 'Space Mono',
                      fontSize: '0.75rem',
                    }}
                    labelStyle={{ color: '#e6e9f0', fontWeight: 600, marginBottom: 4 }}
                    animationDuration={200}
                    formatter={(value) => [`${value}h`]}
                  />
                  <Legend
                    wrapperStyle={{ color: '#c0c7d6', fontSize: '0.75rem', fontFamily: 'Space Mono' }}
                  />
                  <Line
                    type="monotone" dataKey="actual" stroke="#3b82f6"
                    strokeWidth={2} dot={false} name="Actual Delay"
                  />
                  <Line
                    type="monotone" dataKey="predicted" stroke="#00ff88"
                    strokeWidth={2} dot={false} name="Predicted Delay"
                    strokeDasharray="6 3"
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="backtest-metrics">
                <div className="backtest-metric">
                  <span className="backtest-metric-label">RMSE</span>
                  <span className="backtest-metric-value">{backtestData.rmse.toFixed(2)}h</span>
                </div>
                <div className="backtest-metric">
                  <span className="backtest-metric-label">MAE</span>
                  <span className="backtest-metric-value">{backtestData.mae.toFixed(2)}h</span>
                </div>
                <div className="backtest-metric">
                  <span className="backtest-metric-label">Test Set</span>
                  <span className="backtest-metric-value">{backtestData.test_size.toLocaleString()}</span>
                </div>
                <div className="backtest-metric">
                  <span className="backtest-metric-label">Train Set</span>
                  <span className="backtest-metric-value">{backtestData.train_size.toLocaleString()}</span>
                </div>
              </div>
            </div>
          )}

          {/* Interactive supply chain graph */}
          <div className="panel" style={{ marginBottom: '1.5rem' }}>
            <div className="panel-header">
              <span className="panel-title">Supply Chain Network</span>
              <span className="panel-badge">
                {analysisRun ? 'Risk-Colored' : 'Interactive — Drag to Explore'}
              </span>
            </div>
            <p className="chart-subtitle">
              {analysisRun
                ? 'Green = safe, Yellow = warning, Red = critical. Drag to rearrange.'
                : 'Explore the digital twin. Drag nodes to rearrange.'}
            </p>
            {renderInteractiveGraph()}
          </div>

          {/* Start Pipeline button */}
          <div className="run-section">
            <button
              className={`run-btn ${pipelineRunning ? 'loading' : ''} ${analysisRun ? 'completed' : ''}`}
              onClick={handleStartPipeline}
              disabled={pipelineRunning || analysisRun || !isHealthy}
            >
              {pipelineRunning ? (
                <><span className="run-spinner" />Running Pipeline...</>
              ) : analysisRun ? (
                'Pipeline Complete'
              ) : (
                'Start Pipeline'
              )}
            </button>
          </div>

          {/* Pipeline execution log (visible during & after execution) */}
          {pipelineSteps.length > 0 && renderPipelineLog()}
        </div>

        {/* ═══════════════════════════════════════════════
            TAB 2 — ANALYSIS
            Combined: Risk & Forecast + What-If (Monte Carlo)
        ═══════════════════════════════════════════════ */}
        <div className={`tab-content ${activeTab === 'analysis' ? 'active' : ''}`}>
          {!analysisRun ? renderNotRunPlaceholder('Analysis') : (
            <>
              {/* ── Section: Risk & Forecast (ML) ── */}
              <div className="section-heading">
                Risk & Forecast <span className="section-tag">ML — LightGBM</span>
              </div>
              <p className="micro-copy">
                Risk scores are computed per-shipment using a LightGBM classifier trained on historical
                delays, weather severity, and route complexity. Higher scores indicate greater disruption probability.
              </p>

              {/* Estimated Cost of Inaction */}
              <div className="cost-metric">
                <span className="cost-label">Estimated Cost of Inaction</span>
                <span className="cost-value">
                  ${costOfInaction.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </span>
                <span className="cost-desc">
                  Projected loss without mitigation (risk × delay × $1,200/hr cargo cost)
                </span>
              </div>

              {/* Grid with Two Tables: Top 10 Riskiest and Bottom 10 Riskiest */}
              <div className="grid">
                {/* Panel 1: Top 10 Riskiest */}
                <div className="panel">
                  <div className="panel-header">
                    <span className="panel-title">Top 10 Riskiest Shipments</span>
                    <span className="panel-badge">LightGBM</span>
                  </div>
                  <table className="pred-table">
                    <thead>
                      <tr>
                        <th>Trip ID</th>
                        <th>Risk %</th>
                        <th>Expected Delay</th>
                      </tr>
                    </thead>
                    <tbody>
                      {top10.map(p => (
                        <tr key={p.trip_uuid}>
                          <td className="mono">{readableId(p.trip_uuid)}</td>
                          <td>
                            <span className={`risk-badge ${p.risk > 0.7 ? 'high' : p.risk > 0.4 ? 'medium' : 'low'}`}>
                              {(p.risk * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td>{formatDelay(p.expected_delay_hours)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Panel 2: Bottom 10 Safest */}
                <div className="panel">
                  <div className="panel-header">
                    <span className="panel-title">Bottom 10 Riskiest (Safest)</span>
                    <span className="panel-badge">Low Risk</span>
                  </div>
                  <table className="pred-table">
                    <thead>
                      <tr>
                        <th>Trip ID</th>
                        <th>Risk %</th>
                        <th>Expected Delay</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bottom10.map(p => (
                        <tr key={p.trip_uuid}>
                          <td className="mono">{readableId(p.trip_uuid)}</td>
                          <td>
                            <span className="risk-badge low">
                              {(p.risk * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td>{formatDelay(p.expected_delay_hours)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* ── Section: What-If (Monte Carlo) ── */}
              <div className="section-heading">
                What-If Scenarios <span className="section-tag">Monte Carlo — 100 Runs</span>
              </div>
              <p className="micro-copy">
                Each shipment is simulated 100 times with perturbed weather and timing variables.
                The spread between best and worst case reveals supply chain fragility.
              </p>
              {topSimulation && (
                <div className="grid">
                  <div className="panel">
                    <div className="panel-header">
                      <span className="panel-title">Most Volatile: {readableId(topSimulation.trip_uuid)}</span>
                      <span className="panel-badge">100 Scenarios</span>
                    </div>
                    <p className="chart-subtitle">Simulated delay outcomes for the most volatile shipment</p>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={singleSimData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.15)" />
                        <XAxis
                          dataKey="name"
                          tick={{ fill: '#c0c7d6', fontSize: 13, fontWeight: 600, fontFamily: 'Space Mono' }}
                          axisLine={{ stroke: 'rgba(139,146,168,0.2)' }}
                          tickLine={false}
                        />
                        <YAxis
                          tick={{ fill: '#c0c7d6', fontSize: 12 }}
                          label={{ value: 'Delay (hours)', angle: -90, position: 'insideLeft', fill: '#c0c7d6', fontSize: 12 }}
                          axisLine={{ stroke: 'rgba(139,146,168,0.2)' }}
                          tickLine={false}
                        />
                        <Tooltip
                          contentStyle={{
                            background: 'rgba(20, 25, 32, 0.95)',
                            border: '1px solid rgba(139,146,168,0.25)',
                            borderRadius: 8,
                            color: '#e6e9f0',
                            padding: '0.75rem 1rem',
                            boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                            fontFamily: 'Space Mono',
                            fontSize: '0.8rem',
                          }}
                          itemStyle={{ color: '#e6e9f0' }}
                          labelStyle={{ color: '#e6e9f0', fontWeight: 700, marginBottom: 6, fontSize: '0.85rem' }}
                          cursor={{ fill: 'rgba(255,255,255,0.04)', radius: 4 }}
                          animationDuration={200}
                          formatter={(value, name) => [`${value} hours`, 'Simulated Delay']}
                        />
                        <Bar dataKey="delay" radius={[6, 6, 0, 0]} animationDuration={600}>
                          {singleSimData.map((entry, idx) => (
                            <Cell key={idx} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    <p className="whatif-explanation">
                      Across 100 simulated futures, this shipment's delay ranges from{' '}
                      <span className="text-safe">{formatDelay(topSimulation.best_case)}</span> to{' '}
                      <span className="text-critical">{formatDelay(topSimulation.worst_case)}</span>.
                      There is a 10% chance delays exceed{' '}
                      <span className="text-warning">{formatDelay(topSimulation.p90)}</span>.
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ═══════════════════════════════════════════════
            TAB 3 — MITIGATIONS
            Autonomous mitigation with Solana audit trail
        ═══════════════════════════════════════════════ */}
        <div className={`tab-content ${activeTab === 'mitigations' ? 'active' : ''}`}>
          {!analysisRun ? renderNotRunPlaceholder('Mitigations') : (
            <>
              {/* On-chain Summary */}
              <div className="chain-comparison">
                <div className="chain-stat" style={{ width: '100%', borderRight: 'none' }}>
                  <span className="chain-count on-chain">{onChainCount}</span>
                  <span className="chain-label">On-Chain Actions</span>
                  <span className="chain-desc">High-risk strategies recorded on Solana</span>
                </div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <span className="panel-title">Autonomous Mitigations</span>
                  <span className="panel-badge">{mitigationList.length} Shipments</span>
                </div>

                {mitigationList.length > 0 ? mitigationList.map(m => {
                  const story = stories?.[m.trip_uuid];
                  const memoPayload = `ChainReaction|${m.trip_uuid}|risk:${((m.original_risk || 0) * 100).toFixed(0)}%|${m.strategy}`;

                  return (
                    <div key={m.trip_uuid} className="mitigation-card">
                      {/* Header: readable ID + On-Chain badge + Solana link */}
                      <div className="mitigation-header">
                        <div className="mitigation-id-row">
                          <span className="mitigation-trip-id">{readableId(m.trip_uuid)}</span>
                          <span className="threshold-badge on-chain">ON-CHAIN</span>
                        </div>
                        {m.solana_tx && !m.solana_tx.startsWith('DEMO_') ? (
                          <a
                            href={`${SOLANA_EXPLORER}/${m.solana_tx}?cluster=devnet`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="solana-link"
                          >
                            View TX
                          </a>
                        ) : (
                          <span className="solana-demo-badge">On-Chain</span>
                        )}
                      </div>

                      {/* Strategy name */}
                      <div className="mitigation-action">{m.strategy}</div>

                      {/* Risk stats */}
                      <div className="mitigation-desc">
                        Risk: <span className="text-critical">{((m.original_risk || 0) * 100).toFixed(1)}%</span>
                        {' \u2192 '}
                        <span className="text-safe">{(m.mitigated_risk * 100).toFixed(1)}%</span>
                        {' (reduced by '}
                        <span className="text-safe">{(m.expected_risk_reduction * 100).toFixed(1)}%</span>
                        {')'}
                      </div>

                      {/* Embedded journey narrative */}
                      {story && (
                        <div className="embedded-narrative">{story.story}</div>
                      )}

                      {/* Decision Hash Preview */}
                      <div className="decision-hash">
                        <span className="hash-label">Memo Payload</span>
                        <code className="hash-value">{memoPayload}</code>
                      </div>

                      {/* Decision Timeline */}
                      <div className="decision-timeline">
                        <div className="timeline-step completed">
                          <div className="timeline-dot" />
                          <span className="timeline-label">Risk Detected</span>
                          <span className="timeline-time">{pipelineCompletedAt || '--'}</span>
                        </div>
                        <div className="timeline-connector" />
                        <div className="timeline-step completed">
                          <div className="timeline-dot" />
                          <span className="timeline-label">Strategy Selected</span>
                          <span className="timeline-time">{pipelineCompletedAt || '--'}</span>
                        </div>
                        <div className="timeline-connector" />
                        <div className="timeline-step completed">
                          <div className="timeline-dot" />
                          <span className="timeline-label">Recorded On-Chain</span>
                          <span className="timeline-time">{pipelineCompletedAt || '--'}</span>
                        </div>
                      </div>
                    </div>
                  );
                }) : (
                  <p className="loading-text">No on-chain mitigations computed</p>
                )}
              </div>
            </>
          )}
        </div>

      </main>
    </div>
  );
}

export default App;

