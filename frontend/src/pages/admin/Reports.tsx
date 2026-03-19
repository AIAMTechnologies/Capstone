import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import {
  getLeadReport, getLeadGraph, getDealerPerformance, getDealerProjects, getLeadStatusReport,
} from '../../services/api';
import type {
  LeadReportSummary, GraphDataPoint, DealerPerformanceData, DealerProjectData, LeadStatusReport as LeadStatusReportType,
} from '../../types/types';

const TABS = [
  'Dealer Lead Reporting',
  'Dealer Performance',
  'Project Size Breakdown',
  'Lead Status & Conversion',
] as const;
type Tab = typeof TABS[number];

const tabBtnStyle = (active: boolean): React.CSSProperties => ({
  padding: '12px 24px',
  background: 'none',
  border: 'none',
  borderBottom: active ? '3px solid #c91414' : '3px solid transparent',
  color: active ? '#c91414' : '#666',
  fontWeight: active ? 600 : 400,
  cursor: 'pointer',
  fontSize: 14,
});

const thStyle: React.CSSProperties = {
  background: '#f8f9fa', textAlign: 'left' as const, padding: '10px 14px',
  fontSize: 13, fontWeight: 600, color: '#555', borderBottom: '2px solid #ddd',
};
const tdStyle: React.CSSProperties = {
  padding: '10px 14px', borderBottom: '1px solid #eee', fontSize: 13,
};
const cardStyle: React.CSSProperties = {
  background: 'white', borderRadius: 8, padding: 20, boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
};
const sourceSelectStyle: React.CSSProperties = {
  padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14,
};

const Reports: React.FC = () => {
  const [tab, setTab] = useState<Tab>('Dealer Lead Reporting');

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: '#1a1a2e', marginBottom: 20 }}>Reports</h1>

      <div style={{ display: 'flex', gap: 0, borderBottom: '2px solid #eee', marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={tabBtnStyle(tab === t)}>{t}</button>
        ))}
      </div>

      {tab === 'Dealer Lead Reporting' && <OverallSummaryTab />}
      {tab === 'Dealer Performance' && <DealerPerformanceTab />}
      {tab === 'Project Size Breakdown' && <ProjectSizeTab />}
      {tab === 'Lead Status & Conversion' && <LeadStatusTab />}
    </div>
  );
};

// ============================================
// TAB 1: Dealer Lead Reporting (Overall Summary)
// ============================================

const OverallSummaryTab: React.FC = () => {
  const [rows, setRows] = useState<LeadReportSummary[]>([]);
  const [graphData, setGraphData] = useState<GraphDataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError('');
      try {
        const [report, graph] = await Promise.all([getLeadReport(), getLeadGraph(0, 1)]);
        setRows(report.data);
        setGraphData(graph.data);
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load report');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const fmt = (n: number) => n.toLocaleString();

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 600, color: '#1a1a2e', marginBottom: 16 }}>
        Dealer Lead Reporting (Overall Summary)
      </h2>

      {error && <div style={{ color: '#c91414', marginBottom: 12 }}>{error}</div>}
      {loading && <p style={{ color: '#999' }}>Loading...</p>}

      <div style={{ ...cardStyle, marginBottom: 24 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Timeframe</th>
              <th style={thStyle}>Converted</th>
              <th style={thStyle}>Dead</th>
              <th style={thStyle}>Active</th>
              <th style={thStyle}>Total Leads</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.timeframe}>
                <td style={{ ...tdStyle, fontWeight: 600 }}>{r.timeframe}</td>
                <td style={tdStyle}>
                  <span style={{ color: '#1a7a3a', fontWeight: 600 }}>{fmt(r.converted)}</span>
                  <span style={{ color: '#999', marginLeft: 4 }}>({r.converted_pct}%)</span>
                </td>
                <td style={tdStyle}>
                  <span style={{ color: '#c91414', fontWeight: 600 }}>{fmt(r.dead)}</span>
                  <span style={{ color: '#999', marginLeft: 4 }}>({r.dead_pct}%)</span>
                </td>
                <td style={tdStyle}>
                  <span style={{ color: '#1a56db', fontWeight: 600 }}>{fmt(r.active)}</span>
                  <span style={{ color: '#999', marginLeft: 4 }}>({r.active_pct}%)</span>
                </td>
                <td style={{ ...tdStyle, fontWeight: 700 }}>{fmt(r.total_leads)}</td>
              </tr>
            ))}
            {rows.length === 0 && !loading && (
              <tr><td colSpan={5} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No data</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {graphData.length > 0 && (
        <div style={{ ...cardStyle }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Lead Trends (All Time)</h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={graphData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="period"
                fontSize={11}
                tickFormatter={(v: string) => {
                  const d = new Date(v);
                  return d.toLocaleDateString('en-CA', { year: '2-digit', month: 'short' });
                }}
              />
              <YAxis fontSize={12} />
              <Tooltip
                labelFormatter={(v: string) => {
                  const d = new Date(v);
                  return d.toLocaleDateString('en-CA', { year: 'numeric', month: 'long' });
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="total" name="Total" stroke="#1a56db" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="converted" name="Converted" stroke="#1a7a3a" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="dead" name="Dead" stroke="#c91414" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// ============================================
// TAB 2: Dealer Performance (Report 1)
// ============================================

const DealerPerformanceTab: React.FC = () => {
  const [data, setData] = useState<DealerPerformanceData[]>([]);
  const [source, setSource] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sortField, setSortField] = useState<string>('dealer_name');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getDealerPerformance(source || undefined);
      setData(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [source]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('asc');
    }
  };

  const sorted = [...data].sort((a: any, b: any) => {
    const va = a[sortField] ?? '';
    const vb = b[sortField] ?? '';
    if (typeof va === 'number' && typeof vb === 'number') {
      return sortDir === 'asc' ? va - vb : vb - va;
    }
    return sortDir === 'asc'
      ? String(va).localeCompare(String(vb))
      : String(vb).localeCompare(String(va));
  });

  const sortIcon = (field: string) =>
    sortField === field ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 600, color: '#1a1a2e', marginBottom: 4 }}>
        New Report 1 – Dealer Performance (All Lead Sources)
      </h2>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 16 }}>
        Shows {data.length} dealers with Active, Converted, Dead leads and Avg. Response Time.
      </p>

      <div style={{ marginBottom: 16 }}>
        <label style={{ fontSize: 13, fontWeight: 600, color: '#555', marginRight: 8 }}>Filter by Source:</label>
        <select value={source} onChange={e => setSource(e.target.value)} style={sourceSelectStyle}>
          <option value="">All Sources</option>
          <option value="Dealer - Request a Quote or Consultation">Dealer - Request a Quote</option>
          <option value="import">Import</option>
          <option value="website">Website</option>
        </select>
      </div>

      {error && <div style={{ color: '#c91414', marginBottom: 12 }}>{error}</div>}
      {loading && <p style={{ color: '#999' }}>Loading...</p>}

      <div style={cardStyle}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {[
                { key: 'dealer_name', label: 'Dealer Name' },
                { key: 'active_leads', label: 'Active Leads' },
                { key: 'converted', label: 'Converted' },
                { key: 'dead', label: 'Dead' },
                { key: 'total_leads', label: 'Total Leads' },
                { key: 'avg_response_hours', label: 'AVG Response Time' },
              ].map(col => (
                <th
                  key={col.key}
                  style={{ ...thStyle, cursor: 'pointer' }}
                  onClick={() => handleSort(col.key)}
                >
                  {col.label}{sortIcon(col.key)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map(d => (
              <tr key={d.dealer_id}>
                <td style={{ ...tdStyle, fontWeight: 600 }}>{d.dealer_name}</td>
                <td style={{ ...tdStyle, color: '#1a56db' }}>{d.active_leads}</td>
                <td style={{ ...tdStyle, color: '#1a7a3a' }}>{d.converted}</td>
                <td style={{ ...tdStyle, color: '#c91414' }}>{d.dead}</td>
                <td style={{ ...tdStyle, fontWeight: 600 }}>{d.total_leads}</td>
                <td style={tdStyle}>{d.avg_response_str || '-'}</td>
              </tr>
            ))}
            {data.length === 0 && !loading && (
              <tr><td colSpan={6} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No data</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ============================================
// TAB 3: Project Size Breakdown (Report 2)
// ============================================

const ProjectSizeTab: React.FC = () => {
  const [data, setData] = useState<DealerProjectData[]>([]);
  const [source, setSource] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getDealerProjects(source || undefined);
      setData(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [source]);

  useEffect(() => { fetchData(); }, [fetchData]);

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 600, color: '#1a1a2e', marginBottom: 4 }}>
        New Report 2 – Project Size Breakdown by Dealer
      </h2>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 16 }}>
        Breaks down leads by square footage categories for {data.length} dealers.
      </p>

      <div style={{ marginBottom: 16 }}>
        <label style={{ fontSize: 13, fontWeight: 600, color: '#555', marginRight: 8 }}>Filter by Source:</label>
        <select value={source} onChange={e => setSource(e.target.value)} style={sourceSelectStyle}>
          <option value="">All Sources</option>
          <option value="Dealer - Request a Quote or Consultation">Dealer - Request a Quote</option>
          <option value="import">Import</option>
          <option value="website">Website</option>
        </select>
      </div>

      {error && <div style={{ color: '#c91414', marginBottom: 12 }}>{error}</div>}
      {loading && <p style={{ color: '#999' }}>Loading...</p>}

      <div style={{ ...cardStyle, overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 800 }}>
          <thead>
            <tr>
              <th style={thStyle}>Dealer Name</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>1–499 sqft</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>500–999 sqft</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>1,000–3,499 sqft</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>3,500–7,499 sqft</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>7,500–19,999 sqft</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>20,000+ sqft</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>Total</th>
            </tr>
          </thead>
          <tbody>
            {data.map(d => (
              <tr key={d.dealer_id}>
                <td style={{ ...tdStyle, fontWeight: 600 }}>{d.dealer_name}</td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>{d.sqft_1_499 || '-'}</td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>{d.sqft_500_999 || '-'}</td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>{d.sqft_1000_3499 || '-'}</td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>{d.sqft_3500_7499 || '-'}</td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>{d.sqft_7500_19999 || '-'}</td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>{d.sqft_20000_plus || '-'}</td>
                <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 700 }}>{d.total_leads}</td>
              </tr>
            ))}
            {data.length === 0 && !loading && (
              <tr><td colSpan={8} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No data</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ============================================
// TAB 4: Lead Status & Conversion Scores (Report 3)
// ============================================

const LeadStatusTab: React.FC = () => {
  const [data, setData] = useState<LeadStatusReportType[]>([]);
  const [source, setSource] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getLeadStatusReport(source || undefined);
      setData(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [source]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const fmt$ = (v: number | null) => v != null && v > 0 ? `$${Number(v).toLocaleString()}` : '-';

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 600, color: '#1a1a2e', marginBottom: 4 }}>
        New Report 3 – Lead Status & Conversion Scores
      </h2>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 16 }}>
        Shows each dealer's reviewing/undecided, building budget, converted total $, and lead score %.
      </p>

      <div style={{ marginBottom: 16 }}>
        <label style={{ fontSize: 13, fontWeight: 600, color: '#555', marginRight: 8 }}>Filter by Source:</label>
        <select value={source} onChange={e => setSource(e.target.value)} style={sourceSelectStyle}>
          <option value="">All Sources</option>
          <option value="Dealer - Request a Quote or Consultation">Dealer - Request a Quote</option>
          <option value="import">Import</option>
          <option value="website">Website</option>
        </select>
      </div>

      {error && <div style={{ color: '#c91414', marginBottom: 12 }}>{error}</div>}
      {loading && <p style={{ color: '#999' }}>Loading...</p>}

      <div style={cardStyle}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Dealer Name</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>CLIENT Reviewing/Undecided</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>CLIENT Building Budget</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>Converted Total $</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>Lead Score %</th>
            </tr>
          </thead>
          <tbody>
            {data.map(d => (
              <tr key={d.dealer_id}>
                <td style={{ ...tdStyle, fontWeight: 600 }}>{d.dealer_name}</td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>{d.reviewing_undecided}</td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>{d.building_budget}</td>
                <td style={{ ...tdStyle, textAlign: 'center', color: '#1a7a3a', fontWeight: 600 }}>
                  {fmt$(d.converted_total_value)}
                </td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>
                  <span style={{
                    padding: '2px 10px',
                    borderRadius: 12,
                    fontSize: 12,
                    fontWeight: 700,
                    background: d.lead_score_pct >= 30 ? '#dcfce7' : d.lead_score_pct >= 15 ? '#fef3c7' : '#f3f4f6',
                    color: d.lead_score_pct >= 30 ? '#166534' : d.lead_score_pct >= 15 ? '#92400e' : '#666',
                  }}>
                    {d.lead_score_pct}%
                  </span>
                </td>
              </tr>
            ))}
            {data.length === 0 && !loading && (
              <tr><td colSpan={5} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No data</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Reports;
