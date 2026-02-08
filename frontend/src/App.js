import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area } from 'recharts';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';
const SOLANA_EXPLORER = 'https://explorer.solana.com/tx';

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [apiStatus, setApiStatus] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [simulations, setSimulations] = useState(null);
  const [mitigations, setMitigations] = useState(null);
  const [stories, setStories] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predLoading, setPredLoading] = useState(false);
  const [simLoading, setSimLoading] = useState(false);
  const [sortBy, setSortBy] = useState('risk');
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(0);
  const [lastUpdate, setLastUpdate] = useState(0);
  const PAGE_SIZE = 20;

  // Live update timer
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate((prev) => prev + 3);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setApiStatus(response.data);
      setLoading(false);
    } catch (error) {
      setApiStatus({ status: 'error', message: 'Cannot connect to backend' });
      setLoading(false);
    }
  };

  const fetchPredictions = useCallback(async () => {
    setPredLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`);
      const data = response.data;
      if (data.error) { setPredLoading(false); return; }
      const arr = Object.entries(data).map(([trip_uuid, pred]) => ({
        trip_uuid,
        risk: pred.risk,
        expected_delay_hours: pred.expected_delay_hours,
      }));
      setPredictions(arr);
      setPage(0);
    } catch (error) {
      console.error('Failed to fetch predictions:', error);
    }
    setPredLoading(false);
  }, []);

  const fetchSimulations = useCallback(async () => {
    setSimLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/simulate`);
      const data = response.data;
      if (data.error) { setSimLoading(false); return; }
      const arr = Object.entries(data).map(([trip_uuid, sim]) => ({
        trip_uuid,
        ...sim,
      }));
      setSimulations(arr);
    } catch (error) {
      console.error('Failed to fetch simulations:', error);
    }
    setSimLoading(false);
  }, []);

  const fetchMitigations = useCallback(async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/mitigate`);
      if (!response.data.error) setMitigations(response.data);
    } catch (error) {
      console.error('Failed to fetch mitigations:', error);
    }
  }, []);

  const fetchStories = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/story`);
      if (!response.data.error) setStories(response.data);
    } catch (error) {
      console.error('Failed to fetch stories:', error);
    }
  }, []);

  const fetchGraphData = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/graph/viz`);
      if (!response.data.error) setGraphData(response.data);
    } catch (error) {
      console.error('Failed to fetch graph data:', error);
    }
  }, []);

  useEffect(() => {
    if (apiStatus?.shipments_loaded) {
      fetchPredictions();
      fetchSimulations();
      fetchMitigations();
      fetchStories();
      fetchGraphData();
    }
  }, [apiStatus, fetchPredictions, fetchSimulations, fetchMitigations, fetchStories, fetchGraphData]);

  // Sorting
  const sortedPredictions = predictions
    ? [...predictions].sort((a, b) => {
        const mul = sortDir === 'desc' ? -1 : 1;
        return mul * (a[sortBy] - b[sortBy]);
      })
    : [];

  const pagedData = sortedPredictions.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = predictions ? Math.ceil(predictions.length / PAGE_SIZE) : 0;

  const handleSort = (col) => {
    if (sortBy === col) {
      setSortDir(sortDir === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(col);
      setSortDir('desc');
    }
  };

  // Risk chart data: top 30 riskiest
  const chartData = predictions
    ? [...predictions]
        .sort((a, b) => b.risk - a.risk)
        .slice(0, 30)
        .map((p) => ({
          name: p.trip_uuid.slice(-8),
          risk: +(p.risk * 100).toFixed(1),
          delay: +p.expected_delay_hours.toFixed(1),
        }))
    : [];

  const getRiskColor = (risk) => {
    if (risk >= 70) return '#ff3864';
    if (risk >= 40) return '#ffd700';
    return '#00ff88';
  };

  const coloredChartData = chartData.map((entry) => ({
    ...entry,
    fill: getRiskColor(entry.risk),
  }));

  // Simulation chart data: top 20 by worst_case spread
  const simChartData = simulations
    ? [...simulations]
        .sort((a, b) => (b.worst_case - b.best_case) - (a.worst_case - a.best_case))
        .slice(0, 20)
        .map((s) => ({
          name: s.trip_uuid.slice(-8),
          best_case: +s.best_case.toFixed(1),
          expected: +s.expected_delay_hours.toFixed(1),
          worst_case: +s.worst_case.toFixed(1),
          p10: +s.p10.toFixed(1),
          p90: +s.p90.toFixed(1),
        }))
    : [];

  // Selected simulation for detail view (top volatile)
  const topSimulation = simulations
    ? [...simulations].sort((a, b) => (b.worst_case - b.best_case) - (a.worst_case - a.best_case))[0]
    : null;

  // Mitigations: sorted by Solana txs first, then risk reduction
  const mitigationList = mitigations
    ? Object.entries(mitigations)
        .map(([uid, m]) => ({ trip_uuid: uid, ...m }))
        .sort((a, b) => {
          if (a.solana_tx && !b.solana_tx) return -1;
          if (!a.solana_tx && b.solana_tx) return 1;
          return (b.expected_risk_reduction || 0) - (a.expected_risk_reduction || 0);
        })
    : [];

  // Stories list
  const storyList = stories
    ? Object.entries(stories).map(([uid, s]) => ({ trip_uuid: uid, ...s }))
    : [];

  // Summary stats
  const stats = predictions
    ? {
        total: predictions.length,
        highRisk: predictions.filter((p) => p.risk > 0.7).length,
        avgRisk: (predictions.reduce((s, p) => s + p.risk, 0) / predictions.length * 100).toFixed(1),
        avgDelay: (predictions.reduce((s, p) => s + p.expected_delay_hours, 0) / predictions.length).toFixed(1),
        mitigated: mitigationList.length,
        onChain: mitigationList.filter((m) => m.solana_tx).length,
      }
    : null;

  // High risk shipments for overview sidebar
  const highRiskShipments = predictions
    ? [...predictions].sort((a, b) => b.risk - a.risk).slice(0, 10)
    : [];

  // Risk distribution pie chart data
  const riskPieData = predictions
    ? (() => {
        const low = predictions.filter((p) => p.risk <= 0.4).length;
        const medium = predictions.filter((p) => p.risk > 0.4 && p.risk <= 0.7).length;
        const high = predictions.filter((p) => p.risk > 0.7).length;
        return [
          { name: 'Low Risk', value: low, color: '#00ff88' },
          { name: 'Medium Risk', value: medium, color: '#ffd700' },
          { name: 'High Risk', value: high, color: '#ff3864' },
        ];
      })()
    : [];

  // Delay over time line chart: sort predictions by delay and bucket into ranges
  const delayLineData = predictions
    ? (() => {
        const sorted = [...predictions].sort((a, b) => a.expected_delay_hours - b.expected_delay_hours);
        const bucketSize = Math.max(1, Math.floor(sorted.length / 30));
        const buckets = [];
        for (let i = 0; i < sorted.length; i += bucketSize) {
          const slice = sorted.slice(i, i + bucketSize);
          const avgDelay = slice.reduce((s, p) => s + p.expected_delay_hours, 0) / slice.length;
          const avgRisk = slice.reduce((s, p) => s + p.risk, 0) / slice.length;
          buckets.push({
            index: buckets.length + 1,
            delay: +avgDelay.toFixed(2),
            risk: +(avgRisk * 100).toFixed(1),
          });
        }
        return buckets;
      })()
    : [];

  // Delay distribution histogram for Monte Carlo tab
  const delayDistribution = simulations
    ? (() => {
        const delays = simulations.map((s) => s.expected_delay_hours);
        const min = Math.floor(Math.min(...delays));
        const max = Math.ceil(Math.max(...delays));
        const range = max - min || 1;
        const numBins = Math.min(20, range);
        const binWidth = range / numBins;
        const bins = Array.from({ length: numBins }, (_, i) => ({
          range: `${(min + i * binWidth).toFixed(1)}-${(min + (i + 1) * binWidth).toFixed(1)}h`,
          count: 0,
          rangeStart: min + i * binWidth,
        }));
        delays.forEach((d) => {
          const idx = Math.min(numBins - 1, Math.floor((d - min) / binWidth));
          bins[idx].count++;
        });
        return bins;
      })()
    : [];

  // Network graph rendering helper
  const renderNetworkGraph = () => {
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
      return <p className="loading-text">No graph data available</p>;
    }

    const nodes = graphData.nodes;
    const edges = graphData.edges;
    const width = 800;
    const height = 500;
    const padding = 60;

    // Simple force-directed-like layout using circular placement
    const uniqueLabels = [...new Set(nodes.map((n) => n.label))];
    const nodePositions = {};
    nodes.forEach((node, i) => {
      const labelIdx = uniqueLabels.indexOf(node.label);
      const angle = (2 * Math.PI * labelIdx) / uniqueLabels.length;
      const radius = Math.min(width, height) / 2 - padding;
      // Add slight jitter for nodes with same label
      const jitter = (i % 3) * 15;
      nodePositions[node.id] = {
        x: width / 2 + (radius - jitter) * Math.cos(angle),
        y: height / 2 + (radius - jitter) * Math.sin(angle),
      };
    });

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="network-graph">
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="#8b92a8" opacity="0.5" />
          </marker>
        </defs>
        {/* Edges */}
        {edges.slice(0, 100).map((edge, i) => {
          const src = nodePositions[edge.source];
          const tgt = nodePositions[edge.target];
          if (!src || !tgt) return null;
          return (
            <line
              key={`e-${i}`}
              x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
              stroke={edge.color}
              strokeWidth="1"
              opacity="0.3"
              markerEnd="url(#arrowhead)"
            />
          );
        })}
        {/* Nodes */}
        {nodes.slice(0, 60).map((node) => {
          const pos = nodePositions[node.id];
          if (!pos) return null;
          return (
            <g key={node.id}>
              <circle
                cx={pos.x} cy={pos.y} r={6 + node.risk * 10}
                fill={node.color}
                opacity="0.8"
                stroke={node.color}
                strokeWidth="1"
              />
              <text
                x={pos.x} y={pos.y - 12}
                textAnchor="middle"
                fill="#8b92a8"
                fontSize="8"
                fontFamily="Space Mono"
              >
                {node.label}
              </text>
            </g>
          );
        })}
        {/* Legend */}
        <g transform={`translate(${width - 130}, 20)`}>
          <rect x="0" y="0" width="120" height="70" fill="rgba(20,25,32,0.8)" rx="4" />
          <circle cx="15" cy="18" r="5" fill="#00ff88" />
          <text x="28" y="22" fill="#8b92a8" fontSize="9" fontFamily="Space Mono">Low Risk</text>
          <circle cx="15" cy="38" r="5" fill="#ffd700" />
          <text x="28" y="42" fill="#8b92a8" fontSize="9" fontFamily="Space Mono">Medium Risk</text>
          <circle cx="15" cy="58" r="5" fill="#ff3864" />
          <text x="28" y="62" fill="#8b92a8" fontSize="9" fontFamily="Space Mono">High Risk</text>
        </g>
      </svg>
    );
  };

  const getStatusClass = (risk) => {
    if (risk > 0.7) return 'critical';
    if (risk > 0.4) return 'warning';
    return 'safe';
  };

  const isHealthy = apiStatus?.status === 'healthy';

  return (
    <div>
      <div className="grain" />

      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>CHAIN-REACTION</h1>
          <div className="status-bar">
            <div className="status-item">
              <span className={`status-dot ${isHealthy ? 'live' : 'error'}`} />
              <span>{loading ? 'Connecting...' : isHealthy ? 'Live Twin Active' : 'Offline'}</span>
            </div>
            {stats && (
              <div className="status-item">
                <span>{stats.total.toLocaleString()} Trips</span>
              </div>
            )}
            <div className="status-item">
              <span>Last Update: {lastUpdate}s ago</span>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="tabs">
        {['overview', 'predictions', 'simulations', 'mitigation', 'stories'].map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === 'simulations' ? 'Monte Carlo' : tab === 'stories' ? 'Journey Stories' : tab}
          </button>
        ))}
      </div>

      <main className="app-main">

        {/* ===== OVERVIEW TAB ===== */}
        <div className={`tab-content ${activeTab === 'overview' ? 'active' : ''}`}>
          {/* Metric cards */}
          <div className="grid grid-4">
            <div className="metric-card">
              <div className="metric-label">Active Shipments</div>
              <div className={`metric-value ${stats ? 'safe' : ''}`}>
                {stats ? stats.total.toLocaleString() : '--'}
              </div>
              <div className="metric-change">All tracked trips</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">At Risk</div>
              <div className={`metric-value ${stats?.highRisk > 0 ? 'warning' : 'safe'}`}>
                {stats ? stats.highRisk.toLocaleString() : '--'}
              </div>
              <div className="metric-change">Risk &gt; 70%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Critical Alerts</div>
              <div className={`metric-value ${stats?.highRisk > 0 ? 'critical' : 'safe'}`}>
                {stats ? predictions.filter((p) => p.risk > 0.9).length : '--'}
              </div>
              <div className="metric-change">Auto-mitigation active</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Risk Score</div>
              <div className="metric-value blue">
                {stats ? `${stats.avgRisk}%` : '--'}
              </div>
              <div className="metric-change">Across all shipments</div>
            </div>
          </div>

          {/* Risk Pie + Delay Line */}
          <div className="grid grid-2">
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Risk Breakdown</span>
                <span className="panel-badge">All Shipments</span>
              </div>
              {riskPieData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={riskPieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={3}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {riskPieData.map((entry, idx) => (
                        <Cell key={`cell-${idx}`} fill={entry.color} stroke="none" />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ background: '#141920', border: '1px solid rgba(139,146,168,0.12)', borderRadius: 4, color: '#e6e9f0' }}
                      formatter={(value, name) => [`${value} shipments`, name]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="loading-text">No prediction data available</p>
              )}
            </div>

            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Expected Delay Curve</span>
                <span className="panel-badge">All Trips</span>
              </div>
              {delayLineData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={delayLineData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                    <defs>
                      <linearGradient id="delayGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.12)" />
                    <XAxis dataKey="index" tick={{ fill: '#8b92a8', fontSize: 10 }} label={{ value: 'Shipment Bucket', position: 'insideBottom', offset: -5, fill: '#8b92a8', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#8b92a8' }} label={{ value: 'Delay (hours)', angle: -90, position: 'insideLeft', fill: '#8b92a8' }} />
                    <Tooltip
                      contentStyle={{ background: '#141920', border: '1px solid rgba(139,146,168,0.12)', borderRadius: 4, color: '#e6e9f0' }}
                      formatter={(value, name) => name === 'delay' ? [`${value}h`, 'Avg Delay'] : [`${value}%`, 'Avg Risk']}
                    />
                    <Area type="monotone" dataKey="delay" name="delay" stroke="#3b82f6" fill="url(#delayGradient)" strokeWidth={2} />
                    <Line type="monotone" dataKey="risk" name="risk" stroke="#ff3864" strokeWidth={1.5} dot={false} yAxisId={0} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <p className="loading-text">No delay data available</p>
              )}
            </div>
          </div>

          {/* Risk Bar Chart + High-Risk List */}
          <div className="grid grid-2">
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Top Risky Shipments</span>
                <span className="panel-badge">Top 30</span>
              </div>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={coloredChartData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.12)" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" fontSize={10} tick={{ fill: '#8b92a8' }} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#8b92a8' }} label={{ value: 'Risk %', angle: -90, position: 'insideLeft', fill: '#8b92a8' }} />
                    <Tooltip
                      contentStyle={{ background: '#141920', border: '1px solid rgba(139,146,168,0.12)', borderRadius: 4, color: '#e6e9f0' }}
                      formatter={(value, name) => name === 'risk' ? [`${value}%`, 'Risk'] : [`${value}h`, 'Delay']}
                    />
                    <Bar dataKey="risk" name="risk" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="loading-text">No prediction data available</p>
              )}
            </div>

            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">High-Risk Shipments</span>
              </div>
              <div className="shipment-list">
                {highRiskShipments.length > 0 ? highRiskShipments.map((ship) => (
                  <div key={ship.trip_uuid} className={`shipment-item ${getStatusClass(ship.risk)}`}>
                    <div className="shipment-header">
                      <span className="shipment-id">{ship.trip_uuid.slice(-12)}</span>
                      <span className={`risk-badge ${ship.risk > 0.7 ? 'high' : ship.risk > 0.4 ? 'medium' : 'low'}`}>
                        {Math.round(ship.risk * 100)}%
                      </span>
                    </div>
                    <div className="shipment-stats">
                      <span>Delay: {ship.expected_delay_hours.toFixed(1)}h</span>
                      <span>{ship.risk > 0.7 ? 'At Risk' : ship.risk > 0.4 ? 'Warning' : 'On Time'}</span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className={`progress-fill ${getStatusClass(ship.risk)}`}
                        style={{ width: `${Math.min(100, ship.risk * 100)}%` }}
                      />
                    </div>
                  </div>
                )) : (
                  <p className="loading-text">No shipment data</p>
                )}
              </div>
            </div>
          </div>

          {/* Supply Chain Network Graph */}
          <div className="grid">
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Supply Chain Network</span>
                <span className="panel-badge">Color-coded by Risk</span>
              </div>
              <p className="chart-subtitle">Nodes sized by risk score. Green = safe, Yellow = warning, Red = critical</p>
              {renderNetworkGraph()}
            </div>
          </div>

          {/* Top Stories + Mitigation Summary */}
          <div className="grid grid-2">
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Top Journey Stories</span>
                <span className="panel-badge">Top 3</span>
              </div>
              {storyList.length > 0 ? storyList.slice(0, 3).map((s) => (
                <div key={s.trip_uuid} className={`story-card ${s.current_state === 'delayed' ? 'critical' : ''}`}>
                  <div className="story-header">
                    <span className="story-id">{s.trip_uuid.slice(-12)}</span>
                    <span className={`risk-badge ${s.risk > 0.7 ? 'high' : s.risk > 0.4 ? 'medium' : 'low'}`}>
                      {(s.risk * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="story-text">{s.story}</div>
                </div>
              )) : (
                <p className="loading-text">No stories available</p>
              )}
            </div>

            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Mitigation Summary</span>
              </div>
              <div className="mitigation-summary">
                <div className="summary-stat">
                  <span className="summary-label">Total Mitigated</span>
                  <span className="summary-value purple">{mitigationList.length}</span>
                </div>
                <div className="summary-stat">
                  <span className="summary-label">On-Chain Records</span>
                  <span className="summary-value purple">{mitigationList.filter((m) => m.solana_tx).length}</span>
                </div>
                <div className="summary-stat">
                  <span className="summary-label">Avg Risk Reduction</span>
                  <span className="summary-value safe">
                    {mitigationList.length > 0
                      ? `-${(mitigationList.reduce((s, m) => s + m.expected_risk_reduction, 0) / mitigationList.length * 100).toFixed(0)}%`
                      : '--'}
                  </span>
                </div>
              </div>
              {mitigationList.length > 0 && (
                <div className="top-strategies">
                  <div className="panel-header" style={{ marginTop: '1rem' }}>
                    <span className="panel-title">Active Strategies</span>
                  </div>
                  {mitigationList.slice(0, 3).map((m) => (
                    <div key={m.trip_uuid} className="strategy-item">
                      <span className="strategy-id">{m.trip_uuid.slice(-8)}</span>
                      <span className="strategy-action">{m.strategy}</span>
                      <span className="text-safe">-{(m.expected_risk_reduction * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ===== PREDICTIONS TAB ===== */}
        <div className={`tab-content ${activeTab === 'predictions' ? 'active' : ''}`}>
          <div className="grid">
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">ML Model Predictions</span>
                <span className="panel-badge">LightGBM</span>
              </div>
              {predLoading ? (
                <p className="loading-text">Loading predictions...</p>
              ) : predictions ? (
                <>
                  <table className="pred-table">
                    <thead>
                      <tr>
                        <th>Trip UUID</th>
                        <th className="sortable" onClick={() => handleSort('risk')}>
                          Risk {sortBy === 'risk' ? (sortDir === 'desc' ? '\u25BC' : '\u25B2') : ''}
                        </th>
                        <th className="sortable" onClick={() => handleSort('expected_delay_hours')}>
                          Expected Delay {sortBy === 'expected_delay_hours' ? (sortDir === 'desc' ? '\u25BC' : '\u25B2') : ''}
                        </th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pagedData.map((p) => (
                        <tr key={p.trip_uuid}>
                          <td className="mono">{p.trip_uuid}</td>
                          <td>
                            <span className={`risk-badge ${p.risk > 0.7 ? 'high' : p.risk > 0.4 ? 'medium' : 'low'}`}>
                              {(p.risk * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td>{p.expected_delay_hours.toFixed(2)}h</td>
                          <td>
                            <span className={`risk-badge ${p.risk > 0.7 ? 'high' : p.risk > 0.4 ? 'medium' : 'low'}`}>
                              {p.risk > 0.7 ? 'At Risk' : p.risk > 0.4 ? 'Warning' : 'On Time'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="pagination">
                    <button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}>
                      Previous
                    </button>
                    <span>Page {page + 1} of {totalPages}</span>
                    <button onClick={() => setPage(Math.min(totalPages - 1, page + 1))} disabled={page >= totalPages - 1}>
                      Next
                    </button>
                  </div>
                </>
              ) : (
                <p className="loading-text">No prediction data available</p>
              )}
            </div>
          </div>

          {/* Risk bar chart */}
          {chartData.length > 0 && (
            <div className="grid">
              <div className="panel">
                <div className="panel-header">
                  <span className="panel-title">Risk Distribution</span>
                  <span className="panel-badge">Top 30</span>
                </div>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={coloredChartData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.12)" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" fontSize={10} tick={{ fill: '#8b92a8' }} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#8b92a8' }} />
                    <Tooltip
                      contentStyle={{ background: '#141920', border: '1px solid rgba(139,146,168,0.12)', borderRadius: 4, color: '#e6e9f0' }}
                      formatter={(value, name) => name === 'risk' ? [`${value}%`, 'Risk'] : [`${value}h`, 'Delay']}
                    />
                    <Bar dataKey="risk" name="risk" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* ===== MONTE CARLO TAB ===== */}
        <div className={`tab-content ${activeTab === 'simulations' ? 'active' : ''}`}>
          {simLoading ? (
            <div className="panel"><p className="loading-text">Running Monte Carlo simulations...</p></div>
          ) : (
            <>
              {/* Top shipment detail */}
              {topSimulation && (
                <div className="grid">
                  <div className="panel">
                    <div className="panel-header">
                      <span className="panel-title">Monte Carlo Simulation: {topSimulation.trip_uuid.slice(-12)}</span>
                      <span className="panel-badge">{100} Scenarios</span>
                    </div>
                    <p className="chart-subtitle">Most volatile shipment by delay spread</p>
                    <div className="simulation-result">
                      <div className="sim-metric">
                        <div className="sim-label">Best Case</div>
                        <div className="sim-value safe">{topSimulation.best_case.toFixed(1)}h</div>
                      </div>
                      <div className="sim-metric">
                        <div className="sim-label">P10</div>
                        <div className="sim-value blue">{topSimulation.p10.toFixed(1)}h</div>
                      </div>
                      <div className="sim-metric">
                        <div className="sim-label">Expected</div>
                        <div className="sim-value warning">{topSimulation.expected_delay_hours.toFixed(1)}h</div>
                      </div>
                      <div className="sim-metric">
                        <div className="sim-label">P90</div>
                        <div className="sim-value blue">{topSimulation.p90.toFixed(1)}h</div>
                      </div>
                      <div className="sim-metric">
                        <div className="sim-label">Worst Case</div>
                        <div className="sim-value critical">{topSimulation.worst_case.toFixed(1)}h</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* All simulations chart */}
              {simChartData.length > 0 && (
                <div className="grid">
                  <div className="panel">
                    <div className="panel-header">
                      <span className="panel-title">Top 20 Most Volatile Shipments</span>
                    </div>
                    <p className="chart-subtitle">100 scenarios per shipment — best / expected / worst case delays (hours)</p>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={simChartData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.12)" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" fontSize={10} tick={{ fill: '#8b92a8' }} />
                        <YAxis tick={{ fill: '#8b92a8' }} label={{ value: 'Delay (hours)', angle: -90, position: 'insideLeft', fill: '#8b92a8' }} />
                        <Tooltip
                          contentStyle={{ background: '#141920', border: '1px solid rgba(139,146,168,0.12)', borderRadius: 4, color: '#e6e9f0' }}
                          formatter={(value) => [`${value}h`]}
                        />
                        <Legend wrapperStyle={{ color: '#8b92a8' }} />
                        <Bar dataKey="best_case" name="Best Case" fill="#00ff88" radius={[3, 3, 0, 0]} />
                        <Bar dataKey="expected" name="Expected" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                        <Bar dataKey="worst_case" name="Worst Case" fill="#ff3864" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Delay distribution histogram */}
              {delayDistribution.length > 0 && (
                <div className="grid">
                  <div className="panel">
                    <div className="panel-header">
                      <span className="panel-title">Delay Distribution</span>
                      <span className="panel-badge">All Simulations</span>
                    </div>
                    <p className="chart-subtitle">Distribution of expected delays across all simulated shipments</p>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={delayDistribution} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,146,168,0.12)" />
                        <XAxis dataKey="range" angle={-45} textAnchor="end" fontSize={9} tick={{ fill: '#8b92a8' }} />
                        <YAxis tick={{ fill: '#8b92a8' }} label={{ value: 'Count', angle: -90, position: 'insideLeft', fill: '#8b92a8' }} />
                        <Tooltip
                          contentStyle={{ background: '#141920', border: '1px solid rgba(139,146,168,0.12)', borderRadius: 4, color: '#e6e9f0' }}
                          formatter={(value) => [`${value} shipments`, 'Count']}
                        />
                        <Bar dataKey="count" name="Shipments" fill="#a855f7" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ===== MITIGATION TAB ===== */}
        <div className={`tab-content ${activeTab === 'mitigation' ? 'active' : ''}`}>
          <div className="grid grid-4">
            <div className="metric-card">
              <div className="metric-label">Mitigated Shipments</div>
              <div className="metric-value purple">{mitigationList.length}</div>
              <div className="metric-change">Risk &gt; 70%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">On-Chain Records</div>
              <div className="metric-value purple">{mitigationList.filter((m) => m.solana_tx).length}</div>
              <div className="metric-change">Solana devnet</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Risk Reduction</div>
              <div className="metric-value safe">
                {mitigationList.length > 0
                  ? `-${(mitigationList.reduce((s, m) => s + m.expected_risk_reduction, 0) / mitigationList.length * 100).toFixed(0)}%`
                  : '--'}
              </div>
              <div className="metric-change">Per shipment</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Top Strategy</div>
              <div className="metric-value blue" style={{ fontSize: '1rem' }}>
                {mitigationList.length > 0 ? mitigationList[0]?.strategy?.split(' ').slice(0, 2).join(' ') : '--'}
              </div>
              <div className="metric-change">Most applied</div>
            </div>
          </div>

          <div className="grid">
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Active Mitigations</span>
                {mitigationList.filter((m) => m.solana_tx).length > 0 && (
                  <span className="panel-badge">
                    {mitigationList.filter((m) => m.solana_tx).length} on Solana
                  </span>
                )}
              </div>
              {mitigationList.length > 0 ? mitigationList.slice(0, 50).map((m) => (
                <div key={m.trip_uuid} className="mitigation-card">
                  <div className="mitigation-header">
                    <span className="mitigation-action">{m.trip_uuid.slice(-12)}: {m.strategy}</span>
                    {m.solana_tx && !m.solana_tx.startsWith('DEMO_') ? (
                      <a
                        href={`${SOLANA_EXPLORER}/${m.solana_tx}?cluster=devnet`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="solana-link"
                      >
                        View TX
                      </a>
                    ) : m.solana_tx ? (
                      <span className="solana-demo-badge">On-Chain (Demo)</span>
                    ) : null}
                  </div>
                  <div className="mitigation-desc">
                    Risk reduced by {(m.expected_risk_reduction * 100).toFixed(0)}%.
                    Mitigated risk: {(m.mitigated_risk * 100).toFixed(1)}%.
                  </div>
                  <div className="mitigation-stats">
                    <span>Original: <span className="text-critical">{((m.mitigated_risk + m.expected_risk_reduction) * 100).toFixed(0)}%</span></span>
                    <span>After: <span className="text-safe">{(m.mitigated_risk * 100).toFixed(0)}%</span></span>
                    <span>Reduction: <span className="text-safe">-{(m.expected_risk_reduction * 100).toFixed(0)}%</span></span>
                  </div>
                </div>
              )) : (
                <p className="loading-text">No mitigations computed yet</p>
              )}
            </div>
          </div>
        </div>

        {/* ===== JOURNEY STORIES TAB ===== */}
        <div className={`tab-content ${activeTab === 'stories' ? 'active' : ''}`}>
          <div className="grid">
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Shipment Journey Stories</span>
                {storyList.length > 0 && (
                  <span className="panel-badge">{storyList.length} stories</span>
                )}
              </div>
              {storyList.length > 0 ? storyList.map((s) => (
                <div key={s.trip_uuid} className={`story-card ${s.current_state === 'delayed' ? 'critical' : ''}`}>
                  <div className="story-header">
                    <span className="story-id">{s.trip_uuid.slice(-12)}</span>
                    <span className="story-time">
                      {s.previous_state} &rarr; {s.current_state}
                    </span>
                  </div>
                  <div className="story-text">{s.story}</div>
                  <div className="story-meta">
                    <span className={`risk-badge ${s.risk > 0.7 ? 'high' : s.risk > 0.4 ? 'medium' : 'low'}`}>
                      Risk: {(s.risk * 100).toFixed(0)}%
                    </span>
                    <span className="text-muted">Delay: {s.expected_delay_hours.toFixed(1)}h</span>
                    {s.solana_tx && !s.solana_tx.startsWith('DEMO_') ? (
                      <a
                        href={`${SOLANA_EXPLORER}/${s.solana_tx}?cluster=devnet`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="solana-link"
                      >
                        View on Solana
                      </a>
                    ) : s.solana_tx ? (
                      <span className="solana-demo-badge">On-Chain (Demo)</span>
                    ) : (
                      <span className="text-muted">Off-chain</span>
                    )}
                  </div>
                </div>
              )) : (
                <p className="loading-text">No journey stories available yet</p>
              )}
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}

export default App;
