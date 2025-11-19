import React, { useState, useEffect } from 'react';
import { 
  Users, 
  TrendingUp, 
  CheckCircle, 
  XCircle, 
  Clock,
  Award,
  Filter,
  Download,
  Eye,
  User
} from 'lucide-react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { 
  getDashboardStats, 
  getLeads, 
  updateLeadStatus, 
  updateInstallerOverride,
  getHistoricalData 
} from '../services/api';
import { format } from 'date-fns';
import type { DashboardStats, Lead, LeadStatus, HistoricalData, AlternativeInstaller } from '../types';

const COLORS = ['#3498db', '#27ae60', '#e74c3c', '#f39c12'];

type TabType = 'current' | 'historical' | 'profile';

const AdminDashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [leads, setLeads] = useState<Lead[]>([]);
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [historicalLoading, setHistoricalLoading] = useState<boolean>(false);
  const [historicalStatusFilter, setHistoricalStatusFilter] = useState<string>('all');
  const [loading, setLoading] = useState<boolean>(true);
  const [statusFilter, setStatusFilter] = useState<LeadStatus | 'all'>('all');
  const [leadsLoading, setLeadsLoading] = useState<boolean>(false);
  const [selectedLead, setSelectedLead] = useState<Lead | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('current');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const resolveFinalInstallerName = (lead: Lead | null): string => {
    if (!lead) {
      return 'Pending assignment';
    }
    return lead.final_installer_selection || lead.installer_name || lead.assigned_installer_name || 'Pending assignment';
  };

  useEffect(() => {
    loadDashboardData();
  }, [statusFilter]);

  useEffect(() => {
    if (activeTab === 'historical') {
      loadHistoricalData();
    }
  }, [activeTab, historicalStatusFilter]);

  const loadHistoricalData = async () => {
    setHistoricalLoading(true);
    try {
      const response = await getHistoricalData(100, 0, historicalStatusFilter);
      setHistoricalData(response.data);
    } catch (error) {
      console.error('Error loading historical data:', error);
    } finally {
      setHistoricalLoading(false);
    }
  };

  const loadDashboardData = async () => {
    setLoading(true);
    setErrorMessage(null);

    try {
      const statsData = await getDashboardStats();
      setStats(statsData);
    } catch (error) {
      console.error('Error loading dashboard stats:', error);
      setErrorMessage('Unable to load dashboard statistics. Please try again.');
    }

    setLeadsLoading(true);
    try {
      const leadsData = await getLeads(statusFilter === 'all' ? null : statusFilter, 100, 0);
      setLeads(leadsData.leads ?? []);
    } catch (error) {
      console.error('Error loading leads:', error);
      setErrorMessage((prev) => prev ?? 'Unable to load the latest leads. Please try again.');
    } finally {
      setLeadsLoading(false);
    }

    setLoading(false);
  };

  const handleStatusChange = async (leadId: number, newStatus: LeadStatus) => {
    try {
      await updateLeadStatus(leadId, newStatus);
      loadDashboardData();
    } catch (error) {
      console.error('Error updating status:', error);
    }
  };

  const handleInstallerOverride = async (leadId: number, installerId: number | null) => {
    try {
      await updateInstallerOverride(leadId, installerId);
      loadDashboardData();
    } catch (error) {
      console.error('Error updating installer override:', error);
      alert('Failed to update installer assignment');
    }
  };

  const getStatusBadgeClass = (status: LeadStatus): string => {
    const classes = {
      active: 'badge-active',
      converted: 'badge-converted',
      dead: 'badge-dead',
      follow_up: 'badge-follow_up',
    };
    return `badge ${classes[status] || 'badge-active'}`;
  };

  const formatStatus = (status: LeadStatus): string => {
    return status === 'follow_up' ? 'Follow Up' : status.charAt(0).toUpperCase() + status.slice(1);
  };

  if (loading && !stats) {
    return <div className="spinner"></div>;
  }

  // Chart data
  const statusDistribution = stats ? [
    { name: 'Active', value: stats.pending_leads },
    { name: 'Converted', value: stats.completed_leads },
    { name: 'Assigned', value: stats.assigned_leads }
  ] : [];

  //change this to reflect real performance data
  const performanceData = [
    { name: 'Week 1', leads: 12, converted: 5 },
    { name: 'Week 2', leads: 19, converted: 8 },
    { name: 'Week 3', leads: 15, converted: 6 },
    { name: 'Week 4', leads: 22, converted: 11 },
  ];

  return (
    <div className="container" style={{ paddingTop: '40px', paddingBottom: '40px' }}>
      <h1 style={{ marginBottom: '32px', fontSize: '32px', fontWeight: '700' }}>
        Admin Dashboard
      </h1>

      {errorMessage && (
        <div
          style={{
            marginBottom: '24px',
            padding: '16px',
            borderRadius: '8px',
            backgroundColor: '#fdecea',
            color: '#c0392b',
            border: '1px solid #f5b7b1'
          }}
        >
          {errorMessage}
        </div>
      )}

      {/* Tabs */}
      <div style={styles.tabs}>
        <button 
          style={{...styles.tab, ...(activeTab === 'current' ? styles.activeTab : {})}}
          onClick={() => setActiveTab('current')}
        >
          Current Leads
        </button>
        <button 
          style={{...styles.tab, ...(activeTab === 'historical' ? styles.activeTab : {})}}
          onClick={() => setActiveTab('historical')}
        >
          Historical Data
        </button>
        <button 
          style={{...styles.tab, ...(activeTab === 'profile' ? styles.activeTab : {})}}
          onClick={() => setActiveTab('profile')}
        >
          Admin Profile
        </button>
      </div>

      {/* Current Leads Tab */}
      {activeTab === 'current' && (
        <>
          {/* Stats Cards */}
          <div className="grid grid-4" style={{ marginBottom: '40px' }}>
            <div className="stat-card">
              <div style={styles.statIcon}>
                <Users size={32} />
              </div>
              <div className="stat-value">{stats?.total_leads || 0}</div>
              <div className="stat-label">Total Leads</div>
            </div>

            <div className="stat-card secondary">
              <div style={styles.statIcon}>
                <Clock size={32} />
              </div>
              <div className="stat-value">{stats?.pending_leads || 0}</div>
              <div className="stat-label">Active</div>
            </div>

            <div className="stat-card success">
              <div style={styles.statIcon}>
                <CheckCircle size={32} />
              </div>
              <div className="stat-value">{stats?.completed_leads || 0}</div>
              <div className="stat-label">Converted</div>
            </div>

            <div className="stat-card warning">
              <div style={styles.statIcon}>
                <TrendingUp size={32} />
              </div>
              <div className="stat-value">{stats?.conversion_rate || 0}%</div>
              <div className="stat-label">Conversion Rate</div>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-2" style={{ marginBottom: '40px' }}>
            <div className="card">
              <div className="card-header">Lead Status Distribution</div>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={statusDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {statusDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <div className="card-header">Weekly Performance</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="leads" fill="#3498db" name="Total Leads" />
                  <Bar dataKey="converted" fill="#27ae60" name="Converted" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Leads Table */}
          <div className="card">
            <div className="card-header">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                <span>Manage Leads</span>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                  <Filter size={20} />
                  <select
                    className="form-select"
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value as LeadStatus | 'all')}
                    style={{ width: 'auto', padding: '8px 16px' }}
                  >
                    <option value="all">All Leads</option>
                    <option value="active">Active</option>
                    <option value="converted">Converted</option>
                    <option value="dead">Dead</option>
                    <option value="follow_up">Follow Up</option>
                  </select>
                  <button className="btn btn-outline" style={{ padding: '8px 16px' }}>
                    <Download size={20} />
                    Export
                  </button>
                </div>
              </div>
            </div>

            <div className="table-container" style={{ overflowX: 'auto' }}>
              <table className="table" style={{ minWidth: '1400px' }}>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Phone</th>
                    <th>City</th>
                    <th>Job Type</th>
                    <th>Status</th>
                    <th>Installer (ML)</th>
                    <th>Final Installer</th>
                    <th>Score</th>
                    <th style={{ minWidth: '200px' }}>Alternative Options</th>
                    <th>Date</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {leadsLoading ? (
                    <tr>
                      <td colSpan={13} style={{ textAlign: 'center', padding: '32px' }}>
                        <div className="spinner" />
                      </td>
                    </tr>
                  ) : leads.length === 0 ? (
                    <tr>
                      <td colSpan={13} style={{ textAlign: 'center', padding: '32px', color: '#7f8c8d' }}>
                        No leads found for the selected filter.
                      </td>
                    </tr>
                  ) : (
                    leads.map((lead) => (
                      <tr key={lead.id}>
                        <td>#{lead.id}</td>
                        <td style={{ fontWeight: '600' }}>{lead.name}</td>
                        <td>{lead.email}</td>
                        <td>{lead.phone}</td>
                        <td>{lead.city}, {lead.province}</td>
                        <td style={{ textTransform: 'capitalize' }}>{lead.job_type}</td>
                        <td>
                          <span className={getStatusBadgeClass(lead.status)}>
                            {formatStatus(lead.status)}
                          </span>
                        </td>
                        <td>
                          <div style={{ fontSize: '14px' }}>
                            <div style={{ fontWeight: '600', color: '#2c3e50' }}>
                              {lead.installer_name || 'Unassigned'}
                            </div>
                            {lead.installer_city && (
                              <div style={{ fontSize: '12px', color: '#7f8c8d' }}>
                                {lead.installer_city}
                              </div>
                            )}
                          </div>
                        </td>
                        <td>
                          <div style={{ fontSize: '14px' }}>
                            <div style={{ fontWeight: '600', color: '#2c3e50' }}>
                              {resolveFinalInstallerName(lead)}
                            </div>
                            {lead.installer_override_id && (
                              <div style={{ fontSize: '12px', color: '#c0392b' }}>
                                Manual override
                              </div>
                            )}
                          </div>
                        </td>
                        <td>{lead.allocation_score ? lead.allocation_score.toFixed(1) : 'N/A'}</td>
                        <td>
                          {lead.alternative_installers && lead.alternative_installers.length > 0 ? (
                            <select
                              className="form-select"
                              value={lead.installer_override_id || ''}
                              onChange={(e) => {
                                const installerId = e.target.value ? Number(e.target.value) : null;
                                handleInstallerOverride(lead.id, installerId);
                              }}
                              style={{
                                width: '100%',
                                padding: '4px 8px',
                                fontSize: '13px',
                                minWidth: '180px'
                              }}
                              title="Select alternative installer"
                            >
                              <option value="">Other Installers</option>
                              {lead.alternative_installers.map((alt) => (
                                <option key={alt.id} value={alt.id}>
                                  {alt.name} - {alt.city} ({alt.distance_km.toFixed(1)}km, Score: {alt.allocation_score.toFixed(1)})
                                </option>
                              ))}
                            </select>
                          ) : (
                            <span style={{ fontSize: '12px', color: '#95a5a6' }}>
                              No alternatives
                            </span>
                          )}
                        </td>
                        <td>{format(new Date(lead.created_at), 'MMM dd, yyyy')}</td>
                        <td>
                          <div style={{ display: 'flex', gap: '8px' }}>
                            <select
                              className="form-select"
                              value={lead.status}
                              onChange={(e) => handleStatusChange(lead.id, e.target.value as LeadStatus)}
                              style={{ width: 'auto', padding: '4px 8px', fontSize: '14px' }}
                            >
                              <option value="active">Active</option>
                              <option value="converted">Converted</option>
                              <option value="dead">Dead</option>
                              <option value="follow_up">Follow Up</option>
                            </select>
                            <button
                              className="btn btn-outline"
                              style={{ padding: '4px 8px' }}
                              onClick={() => setSelectedLead(lead)}
                            >
                              <Eye size={16} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))
                  ) }
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Historical Data Tab */}
      {activeTab === 'historical' && (
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
              <span>Historical Data Records</span>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <Filter size={20} />
                <select
                  className="form-select"
                  value={historicalStatusFilter}
                  onChange={(e) => setHistoricalStatusFilter(e.target.value)}
                  style={{ width: 'auto', padding: '8px 16px' }}
                >
                  <option value="all">All Status</option>
                  <option value="converted">Converted Sale</option>
                  <option value="New">New</option>
                  <option value="Dead Lead">Dead Lead</option>
                  <option value="Follow Up">Follow Up</option>
                  <option value="Called">Called</option>
                  <option value="Client reviewing">Client reviewing</option>
                </select>
                <button className="btn btn-outline" style={{ padding: '8px 16px' }}>
                  <Download size={20} />
                  Export
                </button>
              </div>
            </div>
          </div>

          {historicalLoading ? (
            <div style={{ padding: '40px', textAlign: 'center' }}>
              <div className="spinner"></div>
            </div>
          ) : (
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Submit Date</th>
                    <th>Name</th>
                    <th>Company</th>
                    <th>City</th>
                    <th>Dealer</th>
                    <th>Final Installer</th>
                    <th>Project Type</th>
                    <th>Status</th>
                    <th>Job Won</th>
                    <th>Value</th>
                    <th>Job Lost</th>
                    <th>Reason</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {historicalData.length === 0 ? (
                    <tr>
                      <td colSpan={14} style={{ textAlign: 'center', padding: '40px', color: '#7f8c8d' }}>
                        No historical data found
                      </td>
                    </tr>
                  ) : (
                    historicalData.map((record) => (
                      <tr key={record.id}>
                        <td>#{record.id}</td>
                        <td>{record.submit_date ? format(new Date(record.submit_date), 'MMM dd, yyyy') : '-'}</td>
                        <td>{record.first_name} {record.last_name}</td>
                        <td>{record.company_name || '-'}</td>
                        <td>{record.city}, {record.province}</td>
                        <td>{record.dealer_name || '-'}</td>
                        <td>{record.final_installer_selection || record.dealer_name || '-'}</td>
                        <td>{record.project_type || '-'}</td>
                        <td>
                          <span className={`badge ${record.current_status === 'converted' ? 'badge-converted' : 'badge-active'}`}>
                            {record.current_status || 'Unknown'}
                          </span>
                        </td>
                        <td>{record.job_won_date ? format(new Date(record.job_won_date), 'MMM dd, yyyy') : '-'}</td>
                        <td>{record.value_of_order ? `$${record.value_of_order.toLocaleString()}` : '-'}</td>
                        <td>{record.job_lost_date ? format(new Date(record.job_lost_date), 'MMM dd, yyyy') : '-'}</td>
                        <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {record.reason || '-'}
                        </td>
                        <td>{format(new Date(record.created_at), 'MMM dd, yyyy')}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}

          <div style={{ padding: '20px', borderTop: '1px solid #e0e0e0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ color: '#7f8c8d' }}>
              Showing {historicalData.length} records
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button className="btn btn-outline" style={{ padding: '8px 16px' }}>
                Previous
              </button>
              <button className="btn btn-outline" style={{ padding: '8px 16px' }}>
                Next
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Profile Tab */}
      {activeTab === 'profile' && (
        <div className="card" style={{ maxWidth: '600px' }}>
          <div className="card-header">Admin Profile</div>
          <div style={{ marginBottom: '24px' }}>
            <div style={styles.profileAvatar}>
              <User size={48} />
            </div>
          </div>
          <div className="form-group">
            <label className="form-label">Username</label>
            <input
              type="text"
              className="form-input"
              value="admin"
              disabled
              style={{ backgroundColor: '#f5f5f5' }}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Email</label>
            <input
              type="email"
              className="form-input"
              value="admin@windowfilmcanada.com"
              disabled
              style={{ backgroundColor: '#f5f5f5' }}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Role</label>
            <input
              type="text"
              className="form-input"
              value="Administrator"
              disabled
              style={{ backgroundColor: '#f5f5f5' }}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Last Login</label>
            <input
              type="text"
              className="form-input"
              value={format(new Date(), 'MMM dd, yyyy HH:mm')}
              disabled
              style={{ backgroundColor: '#f5f5f5' }}
            />
          </div>
        </div>
      )}

      {/* Lead Detail Modal */}
      {selectedLead && (
        <div style={styles.modal} onClick={() => setSelectedLead(null)}>
          <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <h2>Lead Details</h2>
              <button
                onClick={() => setSelectedLead(null)}
                style={styles.closeButton}
              >
                ×
              </button>
            </div>
            <div style={styles.modalBody}>
              <div className="grid grid-2">
                <div>
                  <p style={styles.detailLabel}>Name</p>
                  <p style={styles.detailValue}>{selectedLead.name}</p>
                </div>
                <div>
                  <p style={styles.detailLabel}>Email</p>
                  <p style={styles.detailValue}>{selectedLead.email}</p>
                </div>
                <div>
                  <p style={styles.detailLabel}>Phone</p>
                  <p style={styles.detailValue}>{selectedLead.phone}</p>
                </div>
                <div>
                  <p style={styles.detailLabel}>Job Type</p>
                  <p style={styles.detailValue}>{selectedLead.job_type}</p>
                </div>
                <div>
                  <p style={styles.detailLabel}>Address</p>
                  <p style={styles.detailValue}>
                    {selectedLead.address}<br />
                    {selectedLead.city}, {selectedLead.province} {selectedLead.postal_code}
                  </p>
                </div>
                <div>
                  <p style={styles.detailLabel}>Assigned Installer</p>
                  <p style={styles.detailValue}>
                    {selectedLead.installer_name || 'Unassigned'}
                  </p>
                </div>
                <div>
                  <p style={styles.detailLabel}>Final Installer</p>
                  <p style={styles.detailValue}>
                    {resolveFinalInstallerName(selectedLead)}
                    {selectedLead.installer_override_id && (
                      <span style={{ color: '#c0392b', marginLeft: '6px', fontSize: '13px' }}>
                        (Manual override)
                      </span>
                    )}
                  </p>
                </div>
                <div>
                  <p style={styles.detailLabel}>Allocation Score</p>
                  <p style={styles.detailValue}>
                    {selectedLead.allocation_score ? selectedLead.allocation_score.toFixed(2) : 'N/A'}
                  </p>
                </div>
                <div>
                  <p style={styles.detailLabel}>Distance</p>
                  <p style={styles.detailValue}>
                    {selectedLead.distance_to_installer_km ? 
                      `${selectedLead.distance_to_installer_km.toFixed(1)} km` : 'N/A'}
                  </p>
                </div>
              </div>
              {selectedLead.comments && (
                <div style={{ marginTop: '20px' }}>
                  <p style={styles.detailLabel}>Comments</p>
                  <p style={styles.detailValue}>{selectedLead.comments}</p>
                </div>
              )}
              
              {/* Alternative Installers Section */}
              {selectedLead.alternative_installers && selectedLead.alternative_installers.length > 0 && (
                <div style={{ marginTop: '24px', padding: '16px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
                  <p style={{...styles.detailLabel, marginBottom: '12px'}}>Alternative Installers (Within 50km)</p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {selectedLead.alternative_installers.map((alt) => (
                      <div 
                        key={alt.id} 
                        style={{ 
                          padding: '12px', 
                          backgroundColor: 'white', 
                          borderRadius: '6px',
                          border: '1px solid #e0e0e0'
                        }}
                      >
                        <div style={{ fontWeight: '600', color: '#2c3e50', marginBottom: '4px' }}>
                          {alt.name}
                        </div>
                        <div style={{ fontSize: '13px', color: '#7f8c8d' }}>
                          {alt.city}, {alt.province} • {alt.distance_km.toFixed(1)}km away
                        </div>
                        <div style={{ fontSize: '13px', color: '#7f8c8d', marginTop: '4px' }}>
                          Score: {alt.allocation_score.toFixed(1)} • Active Leads: {alt.active_leads}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const styles = {
  tabs: {
    display: 'flex',
    gap: '8px',
    marginBottom: '32px',
    borderBottom: '2px solid #e0e0e0',
  },
  tab: {
    padding: '12px 24px',
    border: 'none',
    background: 'transparent',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: '600',
    color: '#7f8c8d',
    borderBottom: '3px solid transparent',
    transition: 'all 0.3s',
  },
  activeTab: {
    color: '#c91414',
    borderBottom: '3px solid #c91414',
  },
  statIcon: {
    marginBottom: '12px',
  },
  profileAvatar: {
    width: '120px',
    height: '120px',
    borderRadius: '50%',
    backgroundColor: '#fee',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    margin: '0 auto',
    color: '#c91414',
  },
  modal: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: '12px',
    maxWidth: '800px',
    width: '90%',
    maxHeight: '90vh',
    overflow: 'auto',
  },
  modalHeader: {
    padding: '24px',
    borderBottom: '1px solid #e0e0e0',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  modalBody: {
    padding: '24px',
  },
  closeButton: {
    fontSize: '32px',
    border: 'none',
    background: 'transparent',
    cursor: 'pointer',
    color: '#7f8c8d',
    padding: '0',
    width: '32px',
    height: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  detailLabel: {
    fontSize: '14px',
    color: '#7f8c8d',
    marginBottom: '4px',
    fontWeight: '600',
  },
  detailValue: {
    fontSize: '16px',
    color: '#2c3e50',
    marginBottom: '16px',
  },
};

export default AdminDashboard;