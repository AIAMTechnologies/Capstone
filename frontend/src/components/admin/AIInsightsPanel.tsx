import React, { useState, useEffect, useCallback } from 'react';
import { getAIInsights } from '../../services/api';
import type { AIInsight } from '../../types';

const typeColors: Record<string, { bg: string; border: string; text: string }> = {
  warning: { bg: '#fff8e1', border: '#ffe082', text: '#f57f17' },
  trend: { bg: '#e3f2fd', border: '#90caf9', text: '#1565c0' },
  alert: { bg: '#ffebee', border: '#ef9a9a', text: '#c62828' },
};

const AIInsightsPanel: React.FC = () => {
  const [insights, setInsights] = useState<AIInsight[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  const fetchInsights = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getAIInsights();
      setInsights(result.insights ?? result as any);
    } catch (err: any) {
      const status = err?.response?.status;
      if (status === 404 || status === 501) {
        setError('No AI key configured. AI insights are unavailable.');
      } else {
        setError(err?.response?.data?.detail || 'Failed to load AI insights.');
      }
      setInsights([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchInsights();
  }, [fetchInsights]);

  return (
    <div style={{
      background: 'white',
      borderRadius: 8,
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      marginBottom: 20,
      overflow: 'hidden',
    }}>
      <div
        onClick={() => setCollapsed((c) => !c)}
        style={{
          background: 'linear-gradient(135deg, #1a1a2e, #16213e)',
          color: 'white',
          padding: '12px 20px',
          fontSize: 16,
          fontWeight: 600,
          cursor: 'pointer',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span>AI Insights</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              fetchInsights();
            }}
            style={{
              padding: '4px 12px',
              borderRadius: 4,
              border: '1px solid rgba(255,255,255,0.3)',
              cursor: 'pointer',
              fontSize: 12,
              fontWeight: 500,
              background: 'transparent',
              color: 'white',
            }}
          >
            Refresh
          </button>
          <span style={{ fontSize: 14 }}>{collapsed ? '+' : '-'}</span>
        </div>
      </div>

      {!collapsed && (
        <div style={{ padding: 16 }}>
          {loading ? (
            <div style={{ padding: 20, textAlign: 'center', color: '#7f8c8d' }}>Loading AI insights...</div>
          ) : error ? (
            <div style={{
              padding: 16,
              borderRadius: 6,
              background: '#f8f9fa',
              color: '#7f8c8d',
              textAlign: 'center',
              fontSize: 14,
            }}>
              {error}
            </div>
          ) : insights.length === 0 ? (
            <div style={{ padding: 20, textAlign: 'center', color: '#95a5a6' }}>
              No insights available at this time.
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {insights.map((insight, idx) => {
                const colors = typeColors[insight.type] || typeColors.trend;
                return (
                  <div
                    key={idx}
                    style={{
                      background: colors.bg,
                      border: `1px solid ${colors.border}`,
                      borderRadius: 8,
                      padding: 14,
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                      marginBottom: 6,
                    }}>
                      <span style={{
                        fontWeight: 600,
                        fontSize: 14,
                        color: colors.text,
                      }}>
                        {insight.title}
                      </span>
                      <span style={{
                        fontSize: 10,
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        color: colors.text,
                        opacity: 0.7,
                      }}>
                        {insight.type}
                      </span>
                    </div>
                    <div style={{ fontSize: 13, color: '#444', lineHeight: 1.5, marginBottom: 8 }}>
                      {insight.body}
                    </div>
                    {insight.action && (
                      <div style={{
                        fontSize: 12,
                        fontWeight: 500,
                        color: colors.text,
                        cursor: 'pointer',
                        textDecoration: 'underline',
                      }}>
                        {insight.action}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AIInsightsPanel;
