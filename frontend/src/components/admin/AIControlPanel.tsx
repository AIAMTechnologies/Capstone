import React, { useCallback, useEffect, useState } from 'react';

import { getAIControls, updateAgentEnabled, updateAISpendLimits } from '../../services/api';
import type { AIControlsSnapshot } from '../../types';
import { getApiErrorMessage } from '../../utils/apiErrors';

const inputStyle: React.CSSProperties = { padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14 };
const btnPrimary: React.CSSProperties = { padding: '8px 20px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600, cursor: 'pointer' };

const AIControlPanel: React.FC = () => {
  const [snapshot, setSnapshot] = useState<AIControlsSnapshot | null>(null);
  const [dailyLimit, setDailyLimit] = useState('50');
  const [monthlyLimit, setMonthlyLimit] = useState('500');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const loadControls = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await getAIControls();
      setSnapshot(data);
      setDailyLimit(String(data.daily_spend_limit_cad));
      setMonthlyLimit(String(data.monthly_spend_limit_cad));
    } catch (err) {
      setError(getApiErrorMessage(err, 'Failed to load AI controls.'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadControls();
  }, [loadControls]);

  const handleToggle = async () => {
    if (!snapshot?.can_manage) return;
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const result = await updateAgentEnabled(!snapshot.agent_enabled);
      setSnapshot((current) => current ? { ...current, agent_enabled: result.agent_enabled } : current);
      setSuccess(result.message);
    } catch (err) {
      setError(getApiErrorMessage(err, 'Failed to update AI toggle.'));
    } finally {
      setSaving(false);
    }
  };

  const handleLimitsSave = async () => {
    if (!snapshot?.can_manage) return;
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const result = await updateAISpendLimits({
        daily_spend_limit_cad: Number(dailyLimit),
        monthly_spend_limit_cad: Number(monthlyLimit),
      });
      setSnapshot((current) => current ? {
        ...current,
        daily_spend_limit_cad: result.daily_spend_limit_cad,
        monthly_spend_limit_cad: result.monthly_spend_limit_cad,
      } : current);
      setSuccess(result.message);
    } catch (err) {
      setError(getApiErrorMessage(err, 'Failed to update AI spend limits.'));
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <div style={{ color: '#64748b' }}>Loading AI controls...</div>;
  }

  if (!snapshot) {
    return <div style={{ color: '#64748b' }}>AI controls unavailable.</div>;
  }

  return (
    <div>
      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {success && <div style={{ color: '#1a7a3a', marginBottom: 12, fontSize: 13 }}>{success}</div>}

      <div style={{ marginBottom: 20, padding: 16, borderRadius: 8, background: '#f8f9fa', border: '1px solid #eee' }}>
        <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>Role</div>
        <div style={{ fontSize: 15, fontWeight: 600, color: '#1a1a2e', marginBottom: 12 }}>{snapshot.role}</div>
        <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>AI Agent Status</div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: snapshot.agent_enabled ? '#1a7a3a' : '#c91414' }}>
            {snapshot.agent_enabled ? 'Enabled' : 'Paused'}
          </div>
          <button
            onClick={handleToggle}
            disabled={!snapshot.can_manage || saving}
            style={{ ...btnPrimary, opacity: !snapshot.can_manage || saving ? 0.6 : 1 }}
          >
            {snapshot.agent_enabled ? 'Pause AI' : 'Resume AI'}
          </button>
        </div>
        {!snapshot.can_manage && (
          <div style={{ marginTop: 10, fontSize: 12, color: '#666' }}>
            Superadmin role required to change AI controls.
          </div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: '#555' }}>Daily Spend Limit (CAD)</span>
          <input
            type="number"
            min="0"
            step="0.01"
            value={dailyLimit}
            onChange={(e) => setDailyLimit(e.target.value)}
            disabled={!snapshot.can_manage || saving}
            style={inputStyle}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: '#555' }}>Monthly Spend Limit (CAD)</span>
          <input
            type="number"
            min="0"
            step="0.01"
            value={monthlyLimit}
            onChange={(e) => setMonthlyLimit(e.target.value)}
            disabled={!snapshot.can_manage || saving}
            style={inputStyle}
          />
        </label>
      </div>

      <div style={{ marginTop: 16 }}>
        <button
          onClick={handleLimitsSave}
          disabled={!snapshot.can_manage || saving}
          style={{ ...btnPrimary, opacity: !snapshot.can_manage || saving ? 0.6 : 1 }}
        >
          Save AI Limits
        </button>
      </div>
    </div>
  );
};

export default AIControlPanel;
