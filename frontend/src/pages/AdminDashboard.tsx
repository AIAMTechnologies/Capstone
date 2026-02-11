import React, { useState, useEffect, useRef, useCallback } from 'react';
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
import type { DashboardStats, Lead, LeadStatus, HistoricalData } from '../types';

const COLORS = ['#3498db', '#27ae60', '#e74c3c', '#f39c12'];

type TabType = 'current' | 'historical' | 'profile';

type AdminLeadFormState = Record<string, string>;

const COUNTRY_OPTIONS = ['Canada', 'USA', 'Mexico'];

const COUNTRY_SUBDIVISIONS: Record<string, string[]> = {
  Canada: ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT'],
  USA: ['AL', 'AK', 'AZ', 'CA', 'CO', 'FL', 'GA', 'IL', 'NY', 'TX', 'WA'],
  Mexico: ['CDMX', 'Jalisco', 'Nuevo Leon', 'Puebla', 'Yucatan']
};

const PRODUCT_SERVICE_OPTIONS = [
  'Sun Control',
  'Safety / Security',
  'Graphics - Print/Cut',
  'Privacy/Decorative',
  'Feather Friendly',
  'Automotive'
];

const SQUARE_FOOTAGE_OPTIONS = [
  '1 - 499 sqft',
  '500 - 999 sqft',
  '1000 - 3499 sqft',
  '3500 - 7499 sqft',
  '7500 - 19999 sqft',
  '20000+ sqft'
];

const LEAD_SOURCE_OPTIONS = [
  '3M Canada',
  'National Account',
  'Window Film Canada',
  'Lead 1',
  'Lead 2',
  'Lead 3',
  'Lead 4',
  'Lead 5',
  'PM Expo 2018',
  'TrdMag-1',
  'Tender'
];

const PROJECT_TYPE_OPTIONS = ['Commercial', 'Residential', 'Institutional', 'Hospitality'];

const INITIAL_ADMIN_LEAD_FORM: AdminLeadFormState = {
  first_name: '',
  last_name: '',
  title: '',
  primary_phone: '',
  work_phone: '',
  cell_phone: '',
  email: '',
  company: '',
  address_line_1: '',
  address_line_2: '',
  city: '',
  province: '',
  country: '',
  postal_code: '',
  products_services_1: '',
  products_services_2: '',
  products_services_3: '',
  square_footage: '',
  custom_pick_1: '',
  project_city: '',
  project_type: '',
  business_category: '',
  dealer_email: '',
  other_please_specify: '',
  date_yyyy_mm_dd: '',
  lead_source: '',
  opt_in: '',
  page_name: '',
  url: '',
  variant: '',
  utm_source: '',
  utm_medium: '',
  utm_campaign: '',
  utm_content: '',
  custom_pick_3: ''
};

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
  const [insertLeadError, setInsertLeadError] = useState<string | null>(null);
  const [insertLeadSuccess, setInsertLeadSuccess] = useState<string | null>(null);
  const [adminLeadForm, setAdminLeadForm] = useState<AdminLeadFormState>(INITIAL_ADMIN_LEAD_FORM);
  const [hasInitialized, setHasInitialized] = useState<boolean>(false);
  const leadRequestIdRef = useRef(0);
  const initialStatusFilterRef = useRef<LeadStatus | 'all'>(statusFilter);
  const lastFetchedStatusRef = useRef<LeadStatus | 'all'>(statusFilter);

  const resolveFinalInstallerName = (lead: Lead | null): string => {
    if (!lead) {
      return 'Pending assignment';
    }
    return lead.final_installer_selection || lead.installer_name || lead.assigned_installer_name || 'Pending assignment';
  };

  const loadStats = useCallback(async () => {
    try {
      const statsData = await getDashboardStats();
      setStats(statsData);
    } catch (error) {
      console.error('Error loading dashboard stats:', error);
      setErrorMessage('Unable to load dashboard statistics. Please try again.');
    }
  }, []);

  const loadLeadsForFilter = useCallback(async (filter: LeadStatus | 'all') => {
    const requestId = ++leadRequestIdRef.current;
    setLeadsLoading(true);
    try {
      const leadsData = await getLeads(filter === 'all' ? null : filter, 50, 0);
      if (leadRequestIdRef.current === requestId) {
        setLeads(leadsData.leads ?? []);
      }
    } catch (error) {
      console.error('Error loading leads:', error);
      setErrorMessage((prev) => prev ?? 'Unable to load the latest leads. Please try again.');
      if (leadRequestIdRef.current === requestId) {
        setLeads([]);
      }
    } finally {
      if (leadRequestIdRef.current === requestId) {
        setLeadsLoading(false);
      }
    }
  }, []);

  const refreshLeads = useCallback(async () => {
    await loadLeadsForFilter(statusFilter);
    lastFetchedStatusRef.current = statusFilter;
  }, [statusFilter, loadLeadsForFilter]);

  const loadHistoricalData = useCallback(async () => {
    setHistoricalLoading(true);
    try {
      const response = await getHistoricalData(100, 0, historicalStatusFilter);
      setHistoricalData(response.data);
    } catch (error) {
      console.error('Error loading historical data:', error);
    } finally {
      setHistoricalLoading(false);
    }
  }, [historicalStatusFilter]);

  useEffect(() => {
    if (activeTab === 'historical') {
      loadHistoricalData();
    }
  }, [activeTab, loadHistoricalData]);

  useEffect(() => {
    if (hasInitialized) {
      return;
    }

    let isMounted = true;
    const initializeDashboard = async () => {
      setErrorMessage(null);
      setLoading(true);
      try {
        await Promise.all([loadStats(), loadLeadsForFilter(initialStatusFilterRef.current)]);
        lastFetchedStatusRef.current = initialStatusFilterRef.current;
      } finally {
        if (isMounted) {
          setLoading(false);
          setHasInitialized(true);
        }
      }
    };

    initializeDashboard();

    return () => {
      isMounted = false;
    };
  }, [hasInitialized, loadStats, loadLeadsForFilter]);

  useEffect(() => {
    if (!hasInitialized) {
      return;
    }
    if (lastFetchedStatusRef.current === statusFilter) {
      return;
    }
    refreshLeads();
  }, [statusFilter, hasInitialized, refreshLeads]);

  const handleStatusChange = async (leadId: number, newStatus: LeadStatus) => {
    try {
      await updateLeadStatus(leadId, newStatus);
      await Promise.all([loadStats(), refreshLeads()]);
    } catch (error) {
      console.error('Error updating status:', error);
    }
  };

  const handleInstallerOverride = async (
    leadId: number,
    installerId: number | null,
    installerName?: string | null,
    installerCity?: string | null
  ) => {
    try {
      console.info('Updating installer override', {
        leadId,
        installerId,
        installerName,
        installerCity
      });
      const response = await updateInstallerOverride(leadId, installerId);
      const normalizeFinal = (currentLead: Lead, fallbackName?: string | null) => {
        const responseFinal = (response.final_installer_selection || '').trim();
        if (responseFinal) {
          return responseFinal;
        }
        const preferred = (fallbackName || installerName || '').trim();
        if (preferred) {
          return preferred;
        }
        const responseName = (response.installer_name || '').trim();
        if (responseName) {
          return responseName;
        }
        return resolveFinalInstallerName(currentLead);
      };
      setLeads((prevLeads) =>
        prevLeads.map((lead) =>
          lead.id === leadId
            ? (() => {
                const selectedAlt = installerId
                  ? lead.alternative_installers?.find((alt) => alt.id === installerId)
                  : undefined;
                const fallbackName = installerName || selectedAlt?.name || null;
                const fallbackCity = installerCity || selectedAlt?.city || null;
                return {
                  ...lead,
                  installer_override_id: response.installer_id ?? null,
                  assigned_installer_id:
                    response.assigned_installer_id ?? lead.assigned_installer_id ?? response.installer_id ?? null,
                  installer_name:
                    response.installer_name || fallbackName || lead.installer_name,
                  installer_city:
                    response.installer_city || fallbackCity || lead.installer_city,
                  final_installer_selection: normalizeFinal(
                    lead,
                    fallbackName
                  ),
                };
              })()
            : lead
        )
      );
      setSelectedLead((prev) =>
        prev && prev.id === leadId
          ? (() => {
              const selectedAlt = installerId
                ? prev.alternative_installers?.find((alt) => alt.id === installerId)
                : undefined;
              const fallbackName = installerName || selectedAlt?.name || null;
              const fallbackCity = installerCity || selectedAlt?.city || null;
              return {
                ...prev,
                installer_override_id: response.installer_id ?? null,
                assigned_installer_id:
                  response.assigned_installer_id ?? prev.assigned_installer_id ?? response.installer_id ?? null,
                installer_name:
                  response.installer_name || fallbackName || prev.installer_name,
                installer_city:
                  response.installer_city || fallbackCity || prev.installer_city,
                final_installer_selection: normalizeFinal(
                  prev,
                  fallbackName
                ),
              };
            })()
          : prev
      );
      await refreshLeads();
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

  const handleAdminLeadFieldChange = (field: string, value: string) => {
    setAdminLeadForm((prev) => {
      if (field === 'country') {
        return {
          ...prev,
          country: value,
          province: ''
        };
      }
      return { ...prev, [field]: value };
    });
  };

  const handleInsertLead = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setInsertLeadError(null);
    setInsertLeadSuccess(null);

    if (!adminLeadForm.lead_source) {
      setInsertLeadError('This is a required field. Please select a lead source.');
      return;
    }

    const selectedProducts = [
      adminLeadForm.products_services_1,
      adminLeadForm.products_services_2,
      adminLeadForm.products_services_3
    ].filter(Boolean);

    if (new Set(selectedProducts).size !== selectedProducts.length) {
      setInsertLeadError('Please select different values for Products / Services 1-3.');
      return;
    }

    const createdLead: Lead = {
      id: Date.now(),
      name: `${adminLeadForm.first_name} ${adminLeadForm.last_name}`.trim() || 'Unnamed Lead',
      email: adminLeadForm.email || adminLeadForm.dealer_email || '-',
      phone: adminLeadForm.primary_phone || adminLeadForm.cell_phone || adminLeadForm.work_phone || '-',
      address: [adminLeadForm.address_line_1, adminLeadForm.address_line_2].filter(Boolean).join(', '),
      city: adminLeadForm.city || adminLeadForm.project_city || '-',
      province: adminLeadForm.province || '-',
      postal_code: adminLeadForm.postal_code || undefined,
      job_type: adminLeadForm.project_type.toLowerCase() === 'residential' ? 'residential' : 'commercial',
      comments: [
        adminLeadForm.company && `Company: ${adminLeadForm.company}`,
        adminLeadForm.business_category && `Business Category: ${adminLeadForm.business_category}`,
        adminLeadForm.lead_source && `Lead Source: ${adminLeadForm.lead_source}`,
        adminLeadForm.opt_in && `Opt-In: ${adminLeadForm.opt_in}`
      ]
        .filter(Boolean)
        .join(' | '),
      status: 'active',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    setLeads((prev) => [createdLead, ...prev]);
    setAdminLeadForm(INITIAL_ADMIN_LEAD_FORM);
    setInsertLeadSuccess('Lead inserted into the current list.');
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
          <div className="card" style={{ marginBottom: '28px' }}>
            <form onSubmit={handleInsertLead}>
              <div style={{ backgroundColor: '#c91414', color: '#fff', borderRadius: '8px', padding: '14px 16px', marginBottom: '16px' }}>
                <h2 style={{ fontSize: '24px', marginBottom: '4px' }}>Insert New Lead</h2>
                <p>Please fill in all of the following fields</p>
              </div>

              {insertLeadError && <div className="form-error" style={{ marginBottom: '12px' }}>{insertLeadError}</div>}
              {insertLeadSuccess && <div className="alert alert-success" style={{ marginBottom: '12px' }}>{insertLeadSuccess}</div>}

              <div className="admin-lead-form-grid-3">
                <div>
                  <input className="form-input admin-lead-field" placeholder="First Name" value={adminLeadForm.first_name} onChange={(e) => handleAdminLeadFieldChange('first_name', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="Last Name" value={adminLeadForm.last_name} onChange={(e) => handleAdminLeadFieldChange('last_name', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="Title" value={adminLeadForm.title} onChange={(e) => handleAdminLeadFieldChange('title', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="Primary Phone" value={adminLeadForm.primary_phone} onChange={(e) => handleAdminLeadFieldChange('primary_phone', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="Work Phone" value={adminLeadForm.work_phone} onChange={(e) => handleAdminLeadFieldChange('work_phone', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="Cell Phone" value={adminLeadForm.cell_phone} onChange={(e) => handleAdminLeadFieldChange('cell_phone', e.target.value)} />
                  <input type="email" className="form-input admin-lead-field" placeholder="Email" value={adminLeadForm.email} onChange={(e) => handleAdminLeadFieldChange('email', e.target.value)} />
                </div>

                <div>
                  <input className="form-input admin-lead-field" placeholder="Company" value={adminLeadForm.company} onChange={(e) => handleAdminLeadFieldChange('company', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="Address Line 1" value={adminLeadForm.address_line_1} onChange={(e) => handleAdminLeadFieldChange('address_line_1', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="Address Line 2" value={adminLeadForm.address_line_2} onChange={(e) => handleAdminLeadFieldChange('address_line_2', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="City" value={adminLeadForm.city} onChange={(e) => handleAdminLeadFieldChange('city', e.target.value)} />
                  <select className="form-select admin-lead-field" value={adminLeadForm.province} onChange={(e) => handleAdminLeadFieldChange('province', e.target.value)}>
                    <option value="">Province</option>
                    {(COUNTRY_SUBDIVISIONS[adminLeadForm.country] || []).map((item) => (
                      <option key={item} value={item}>{item}</option>
                    ))}
                  </select>
                  <select className="form-select admin-lead-field" value={adminLeadForm.country} onChange={(e) => handleAdminLeadFieldChange('country', e.target.value)}>
                    <option value="">Country</option>
                    {COUNTRY_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                  <input className="form-input admin-lead-field" placeholder="Postal Code" value={adminLeadForm.postal_code} onChange={(e) => handleAdminLeadFieldChange('postal_code', e.target.value)} />
                </div>

                <div>
                  <select className="form-select admin-lead-field" value={adminLeadForm.products_services_1} onChange={(e) => handleAdminLeadFieldChange('products_services_1', e.target.value)}>
                    <option value="">Products / Services 1</option>
                    {PRODUCT_SERVICE_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                  <select className="form-select admin-lead-field" value={adminLeadForm.products_services_2} onChange={(e) => handleAdminLeadFieldChange('products_services_2', e.target.value)}>
                    <option value="">Products / Services 2</option>
                    {PRODUCT_SERVICE_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                  <select className="form-select admin-lead-field" value={adminLeadForm.products_services_3} onChange={(e) => handleAdminLeadFieldChange('products_services_3', e.target.value)}>
                    <option value="">Products / Services 3</option>
                    {PRODUCT_SERVICE_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                  <select className="form-select admin-lead-field" value={adminLeadForm.square_footage} onChange={(e) => handleAdminLeadFieldChange('square_footage', e.target.value)}>
                    <option value="">Square Footage</option>
                    {SQUARE_FOOTAGE_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                  <input className="form-input admin-lead-field" placeholder="Custom Pick 1" value={adminLeadForm.custom_pick_1} onChange={(e) => handleAdminLeadFieldChange('custom_pick_1', e.target.value)} />
                  <input className="form-input admin-lead-field" placeholder="Project City" value={adminLeadForm.project_city} onChange={(e) => handleAdminLeadFieldChange('project_city', e.target.value)} />
                  <select className="form-select admin-lead-field" value={adminLeadForm.project_type} onChange={(e) => handleAdminLeadFieldChange('project_type', e.target.value)}>
                    <option value="">Project Type</option>
                    {PROJECT_TYPE_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                </div>
              </div>

              <div className="admin-lead-form-grid-3">
                <div>
                  <input className="form-input admin-lead-field" placeholder="Business Category" value={adminLeadForm.business_category} onChange={(e) => handleAdminLeadFieldChange('business_category', e.target.value)} />
                  <input type="email" className="form-input admin-lead-field" placeholder="Dealer Email" value={adminLeadForm.dealer_email} onChange={(e) => handleAdminLeadFieldChange('dealer_email', e.target.value)} />
                </div>
                <div>
                  <input className="form-input admin-lead-field" placeholder="Other Please Specify" value={adminLeadForm.other_please_specify} onChange={(e) => handleAdminLeadFieldChange('other_please_specify', e.target.value)} />
                  <input type="date" className="form-input admin-lead-field" value={adminLeadForm.date_yyyy_mm_dd} onChange={(e) => handleAdminLeadFieldChange('date_yyyy_mm_dd', e.target.value)} />
                </div>
                <div>
                  <select className="form-select admin-lead-field" value={adminLeadForm.lead_source} onChange={(e) => handleAdminLeadFieldChange('lead_source', e.target.value)}>
                    <option value="">Lead Source *</option>
                    {LEAD_SOURCE_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                  <select className="form-select admin-lead-field" value={adminLeadForm.opt_in} onChange={(e) => handleAdminLeadFieldChange('opt_in', e.target.value)}>
                    <option value="">Opt-In</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
              </div>

              <div className="admin-lead-form-grid-6">
                <input className="form-input admin-lead-field admin-span-3" placeholder="Page Name" value={adminLeadForm.page_name} onChange={(e) => handleAdminLeadFieldChange('page_name', e.target.value)} />
                <input className="form-input admin-lead-field admin-span-3" placeholder="URL" value={adminLeadForm.url} onChange={(e) => handleAdminLeadFieldChange('url', e.target.value)} />
                <input className="form-input admin-lead-field admin-span-2" placeholder="Variant" value={adminLeadForm.variant} onChange={(e) => handleAdminLeadFieldChange('variant', e.target.value)} />
                <input className="form-input admin-lead-field admin-span-2" placeholder="UTM Source" value={adminLeadForm.utm_source} onChange={(e) => handleAdminLeadFieldChange('utm_source', e.target.value)} />
                <input className="form-input admin-lead-field admin-span-2" placeholder="UTM Medium" value={adminLeadForm.utm_medium} onChange={(e) => handleAdminLeadFieldChange('utm_medium', e.target.value)} />
                <input className="form-input admin-lead-field admin-span-2" placeholder="UTM Campaign" value={adminLeadForm.utm_campaign} onChange={(e) => handleAdminLeadFieldChange('utm_campaign', e.target.value)} />
                <input className="form-input admin-lead-field admin-span-2" placeholder="UTM Content" value={adminLeadForm.utm_content} onChange={(e) => handleAdminLeadFieldChange('utm_content', e.target.value)} />
                <input className="form-input admin-lead-field admin-span-2" placeholder="Custom Pick 3" value={adminLeadForm.custom_pick_3} onChange={(e) => handleAdminLeadFieldChange('custom_pick_3', e.target.value)} />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '16px' }}>
                <button type="submit" className="btn btn-primary">Insert New Lead</button>
              </div>
            </form>
          </div>

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
                              value={lead.installer_override_id ? String(lead.installer_override_id) : ''}
                              onChange={(e) => {
                                const { value } = e.target;
                                const installerId = value ? Number(value) : null;
                                const selectedAlt = installerId
                                  ? lead.alternative_installers?.find((alt) => alt.id === installerId)
                                  : undefined;
                                handleInstallerOverride(
                                  lead.id,
                                  installerId,
                                  selectedAlt?.name ?? null,
                                  selectedAlt?.city ?? null
                                );
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
                                <option
                                  key={alt.id}
                                  value={String(alt.id)}
                                >
                                  {[
                                    alt.name,
                                    [alt.city, alt.province].filter(Boolean).join(', '),
                                    `${alt.distance_km.toFixed(1)} km`,
                                    `Score ${alt.allocation_score.toFixed(2)}`,
                                  ]
                                    .filter(Boolean)
                                    .join(' - ')}
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
