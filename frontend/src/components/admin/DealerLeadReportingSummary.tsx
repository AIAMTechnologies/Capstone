import React, { useCallback, useEffect, useState } from 'react';
import { getLeadReport } from '../../services/api';
import type { LeadReportSummary } from '../../types/types';
import { getApiErrorMessage } from '../../utils/apiErrors';

interface DealerLeadReportingSummaryProps {
  refreshToken?: number;
}

const cardStyle: React.CSSProperties = {
  background: 'white',
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  marginBottom: 20,
  overflow: 'hidden',
};

const headerStyle: React.CSSProperties = {
  background: '#c91414',
  color: 'white',
  padding: '12px 20px',
  fontSize: 16,
  fontWeight: 600,
};

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '10px 12px',
  fontSize: 12,
  fontWeight: 700,
  color: '#4b5563',
  borderBottom: '2px solid #e5e7eb',
  textTransform: 'uppercase',
  letterSpacing: 0.4,
};

const tdStyle: React.CSSProperties = {
  padding: '10px 12px',
  borderBottom: '1px solid #eef2f7',
  fontSize: 13,
  color: '#374151',
};

const fmtPair = (count: number, pct: number) => `${count} ~ ${pct.toFixed(1)}%`;

const DealerLeadReportingSummary: React.FC<DealerLeadReportingSummaryProps> = ({ refreshToken = 0 }) => {
  const [rows, setRows] = useState<LeadReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadRows = useCallback(async () => {
    setLoading(true);
    try {
      const response = await getLeadReport();
      setRows(response.data);
      setError(null);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load dealer lead reporting.'));
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRows();
  }, [loadRows, refreshToken]);

  return (
    <div style={cardStyle}>
      <div style={headerStyle}>Dealer Lead Reporting</div>
      <div style={{ padding: 16 }}>
        {loading ? (
          <div style={{ color: '#7f8c8d' }}>Loading dealer lead reporting...</div>
        ) : error ? (
          <div style={{ padding: 12, borderRadius: 6, background: '#fdecea', color: '#c0392b' }}>{error}</div>
        ) : (
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
              {rows.map((row) => (
                <tr key={row.timeframe}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{row.timeframe}</td>
                  <td style={tdStyle}>{fmtPair(row.converted, row.converted_pct)}</td>
                  <td style={tdStyle}>{fmtPair(row.dead, row.dead_pct)}</td>
                  <td style={tdStyle}>{fmtPair(row.active, row.active_pct)}</td>
                  <td style={{ ...tdStyle, fontWeight: 700 }}>{row.total_leads}</td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#95a5a6' }} colSpan={5}>
                    No dealer lead reporting rows available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default DealerLeadReportingSummary;
