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
  Eye
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
import { getDashboardStats, getLeads, updateLeadStatus } from '../services/api';
import { format } from 'date-fns';
import type { DashboardStats, Lead, LeadStatus } from '../types';

const COLORS = ['#3498db', '#27ae60', '#e74c3c', '#f39c12'];

type TabType = 'current' | 'historical' | 'profile';

const AdminDashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [leads, setLeads] = useState<Lead[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [statusFilter, setStatusFilter] = useState<LeadStatus | 'all'>('active');
  const [selectedLead, setSelectedLead] = useState<Lead | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('current');

  useEffect(() => {
    loadDashboardData();
  }, [statusFilter]);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [statsData, leadsData] = await Promise.all([
        getDashboardStats(),
        getLeads(statusFilter === 'all' ? null : statusFilter, 100, 0)
      ]);
      
      setStats(statsData);
      setLeads(leadsData.leads);
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStatusChange = async (leadId, newStatus) => {
    try {
      await updateLeadStatus(leadId, newStatus);
      loadDashboardData();
    } catch (error) {
      console.error('Error updating status:', error);
    }
  };

  const getStatusBadgeClass = (status: LeadStatus): string => {
    const classes = {
      active: 'badge-active',
      converted: 'badge-converted',
      dead: 'badge-dead'
    };
    return `badge ${classes[status] || 'badge-active'}`;
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
              <div className="stat-label">Pending</div>
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
                  </select>
                  <button className="btn btn-outline" style={{ padding: '8px 16px' }}>
                    <Download size={20} />
                    Export
                  </button>
                </div>
              </div>
            </div>

            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Phone</th>
                    <th>City</th>
                    <th>Job Type</th>
                    <th>Status</th>
                    <th>Installer</th>
                    <th>Score</th>
                    <th>Date</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {leads.map((lead) => (
                    <tr key={lead.id}>
                      <td>#{lead.id}</td>
                      <td style={{ fontWeight: '600' }}>{lead.name}</td>
                      <td>{lead.email}</td>
                      <td>{lead.phone}</td>
                      <td>{lead.city}, {lead.province}</td>
                      <td style={{ textTransform: 'capitalize' }}>{lead.job_type}</td>
                      <td>
                        <span className={getStatusBadgeClass(lead.status)}>
                          {lead.status}
                        </span>
                      </td>
                      <td>{lead.installer_name || 'Unassigned'}</td>
                      <td>{lead.allocation_score ? lead.allocation_score.toFixed(1) : 'N/A'}</td>
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
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Historical Data Tab */}
      {activeTab === 'historical' && (
        <div className="card">
          <div className="card-header">Historical Lead Data</div>
          <div style={{ marginBottom: '32px' }}>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="leads" stroke="#3498db" strokeWidth={2} name="Total Leads" />
                <Line type="monotone" dataKey="converted" stroke="#27ae60" strokeWidth={2} name="Converted" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-3">
            <div className="card">
              <h3 style={{ marginBottom: '16px' }}>Last 30 Days</h3>
              <div className="stat-value" style={{ color: '#3498db' }}>87</div>
              <p style={{ color: '#7f8c8d' }}>Total Leads</p>
            </div>
            <div className="card">
              <h3 style={{ marginBottom: '16px' }}>Last 90 Days</h3>
              <div className="stat-value" style={{ color: '#27ae60' }}>243</div>
              <p style={{ color: '#7f8c8d' }}>Total Leads</p>
            </div>
            <div className="card">
              <h3 style={{ marginBottom: '16px' }}>This Year</h3>
              <div className="stat-value" style={{ color: '#f39c12' }}>1,247</div>
              <p style={{ color: '#7f8c8d' }}>Total Leads</p>
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
    position: 'fixed',
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