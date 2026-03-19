import axios, { AxiosError } from 'axios';
import { env } from '../config/env';
import type {
  LeadFormData,
  LoginResponse,
  DashboardStats,
  LeadsResponse,
  Lead,
  LeadStatus,
  Installer,
  HistoricalDataResponse,
  ExtendedLead,
  Dealer,
  LeadLog,
  LeadReportSummary,
  GraphDataPoint,
  DealerPerformanceData,
  DealerProjectData,
  LeadStatusReport,
  ResponseTimeData,
  ResourcePage,
  DealerNotificationStatus,
  AILeadScore,
  AIMatchExplanation,
  AIInsight,
  AIEmailDraft,
  AIEnrichment,
  AIConversionPrediction,
  AIChurnRisk,
  EmailSyncConfig,
  EmailMessage,
  ClosureReview,
  EmailLeadContext,
  EmailSyncStatus,
  EmailSyncResult,
} from '../types';

const API_BASE_URL = env.apiUrl || 'http://localhost:8000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors globally
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      if (window.location.pathname !== '/admin/login') {
        window.location.href = '/admin/login';
      }
    }
    return Promise.reject(error);
  }
);

// ============================================
// PUBLIC API - Contact Form
// ============================================

export const submitLead = async (leadData: LeadFormData): Promise<Lead> => {
  const response = await api.post<Lead>('/leads', leadData);
  return response.data;
};

export const getPublicGoogleMapsApiKey = async (): Promise<string> => {
  const response = await api.get<{ googleMapsApiKey?: string }>('/config/map-key');
  return (response.data.googleMapsApiKey ?? '').trim();
};

// ============================================
// ADMIN AUTHENTICATION
// ============================================

export const login = async (username: string, password: string): Promise<LoginResponse> => {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);
  
  const response = await axios.post<LoginResponse>(
    `${API_BASE_URL}/admin/login`, 
    formData,
    {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    }
  );
  
  return response.data;
};

// ============================================
// ADMIN DASHBOARD APIs
// ============================================

export const getDashboardStats = async (): Promise<DashboardStats> => {
  const response = await api.get<DashboardStats>('/admin/dashboard');
  return response.data;
};

export const getLeads = async (
  status: LeadStatus | 'all' | null = null, 
  limit = 50, 
  offset = 0
): Promise<LeadsResponse> => {
  const params: any = { limit, offset };
  if (status && status !== 'all') {
    params.status = status;
  }
  
  const response = await api.get<LeadsResponse>('/admin/leads', { params });
  return response.data;
};

export const getLeadDetail = async (leadId: number): Promise<Lead> => {
  const response = await api.get<Lead>(`/admin/leads/${leadId}`);
  return response.data;
};

export const updateLeadStatus = async (
  leadId: number, 
  status: LeadStatus
): Promise<{ message: string; lead_id: number; new_status: string }> => {
  const response = await api.patch(`/admin/leads/${leadId}/status`, null, {
    params: { status }
  });
  return response.data;
};

export const updateInstallerOverride = async (
  leadId: number,
  installerId: number | null
): Promise<{
  message: string;
  lead_id: number;
  installer_id: number | null;
  assigned_installer_id?: number | null;
  final_installer_selection?: string;
  installer_name?: string | null;
  installer_city?: string | null;
}> => {
  const response = await api.patch(`/admin/leads/${leadId}/installer-override`, {
    installer_id: installerId
  });
  return response.data;
};

export const getInstallers = async (): Promise<{ installers: Installer[]; count: number }> => {
  const response = await api.get('/admin/installers');
  return response.data;
};

export const getHistoricalData = async (
  limit = 100,
  offset = 0,
  status?: string
): Promise<HistoricalDataResponse> => {
  const params: any = { limit, offset };
  if (status && status !== 'all') {
    params.status = status;
  }
  
  const response = await api.get<HistoricalDataResponse>('/admin/historical-data', { params });
  return response.data;
};

// ============================================
// ADMIN LEAD MANAGEMENT (new routes)
// ============================================

export const getUnassignedLeads = async (): Promise<{ leads: ExtendedLead[]; count: number }> => {
  const response = await api.get('/admin/unassigned-leads');
  return response.data;
};

export const getActiveLeads = async (): Promise<{ leads: ExtendedLead[]; count: number }> => {
  const response = await api.get('/admin/active-leads');
  return response.data;
};

export const createNewLead = async (leadData: Record<string, any>): Promise<{ message: string; lead_id: number }> => {
  const response = await api.post('/admin/leads/new', leadData);
  return response.data;
};

export const assignDealerToLead = async (leadId: number, dealerIds: number[]): Promise<{ message: string }> => {
  const response = await api.post(`/admin/leads/${leadId}/assign-dealer`, { dealer_ids: dealerIds });
  return response.data;
};

export const updateLead = async (leadId: number, data: Record<string, any>): Promise<{ message: string }> => {
  const response = await api.post(`/admin/leads/${leadId}/update`, data);
  return response.data;
};

export const archiveLead = async (leadId: number): Promise<{ message: string }> => {
  const response = await api.post(`/admin/leads/${leadId}/archive`);
  return response.data;
};

export const deleteLead = async (leadId: number): Promise<{ message: string }> => {
  const response = await api.post(`/admin/leads/${leadId}/delete`);
  return response.data;
};

// ============================================
// DEALER API
// ============================================

export const getDealers = async (province?: string): Promise<{ dealers: Dealer[]; count: number }> => {
  const params: any = {};
  if (province) params.province = province;
  const response = await api.get('/admin/dealers', { params });
  return response.data;
};

export const getDealerOptions = async (): Promise<{ dealers: Dealer[] }> => {
  const response = await api.get('/admin/dealers/options');
  return response.data;
};

// ============================================
// LEAD LOGS
// ============================================

export const getLeadLogs = async (leadId: number): Promise<{ logs: LeadLog[]; count: number }> => {
  const response = await api.get('/admin/lead-log', { params: { lead_id: leadId } });
  return response.data;
};

export const createLeadLog = async (data: { lead_id: number; log_type: string; message: string; dealer_id?: number }): Promise<{ message: string }> => {
  const response = await api.post('/admin/lead-log', data);
  return response.data;
};

// ============================================
// HISTORY
// ============================================

export const getHistoryLeads = async (params: {
  start_date?: string;
  end_date?: string;
  province?: string;
  dealer?: string;
  limit?: number;
  offset?: number;
}): Promise<{ leads: ExtendedLead[]; count: number; total: number }> => {
  const response = await api.get('/admin/history-leads', { params });
  return response.data;
};

export const updateValueOfOrder = async (leadId: number, value: number): Promise<{ message: string }> => {
  const response = await api.post(`/admin/leads/${leadId}/update-value`, null, { params: { value } });
  return response.data;
};

// ============================================
// REPORTS
// ============================================

export const getLeadReport = async (): Promise<{ data: LeadReportSummary[] }> => {
  const response = await api.get('/admin/reports/lead-report');
  return response.data;
};

export const getLeadGraph = async (tf: number = 6, opt: number = 1): Promise<{ data: GraphDataPoint[] }> => {
  const response = await api.get('/admin/reports/lead-graph', { params: { tf, opt } });
  return response.data;
};

export const getDealerPerformance = async (source?: string): Promise<{ data: any[] }> => {
  const params: any = {};
  if (source) params.source = source;
  const response = await api.get('/admin/reports/dealer-performance', { params });
  return response.data;
};

export const getDealerProjects = async (source?: string): Promise<{ data: DealerProjectData[] }> => {
  const params: any = {};
  if (source) params.source = source;
  const response = await api.get('/admin/reports/dealer-projects', { params });
  return response.data;
};

export const getLeadStatusReport = async (source?: string): Promise<{ data: LeadStatusReport[] }> => {
  const params: any = {};
  if (source) params.source = source;
  const response = await api.get('/admin/reports/lead-status', { params });
  return response.data;
};

// Legacy compat
export const getResponseTimes = getDealerPerformance;

// ============================================
// RESOURCES
// ============================================

export const getResources = async (): Promise<{ pages: ResourcePage[] }> => {
  const response = await api.get('/admin/resources');
  return response.data;
};

export const getResource = async (pageId: number): Promise<ResourcePage> => {
  const response = await api.get(`/admin/resources/${pageId}`);
  return response.data;
};

export const createResource = async (data: { title: string; slug?: string; content?: string; parent_id?: number; sort_order?: number }): Promise<{ message: string; id: number }> => {
  const response = await api.post('/admin/resources', data);
  return response.data;
};

export const updateResource = async (pageId: number, data: Record<string, any>): Promise<{ message: string }> => {
  const response = await api.put(`/admin/resources/${pageId}`, data);
  return response.data;
};

export const deleteResource = async (pageId: number): Promise<{ message: string }> => {
  const response = await api.delete(`/admin/resources/${pageId}`);
  return response.data;
};

// ============================================
// TOOLS
// ============================================

export const uploadCSV = async (file: File): Promise<{ headers: string[]; preview: Record<string, string>[]; total_rows: number }> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/admin/tools/csv-upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000,
  });
  return response.data;
};

export const importCSV = async (data: { mappings: { csv_column: string; db_column: string }[]; data: Record<string, string>[] }): Promise<{ message: string; inserted: number; errors: string[] }> => {
  const response = await api.post('/admin/tools/csv-import', data);
  return response.data;
};

export const sendMassEmail = async (data: { dealer_ids: number[]; subject: string; body: string }): Promise<{ message: string }> => {
  const response = await api.post('/admin/tools/mass-email', data);
  return response.data;
};

export const getNotificationCheck = async (): Promise<{ dealers: DealerNotificationStatus[] }> => {
  const response = await api.get('/admin/tools/notification-check');
  return response.data;
};

export const resendNotification = async (dealerId: number): Promise<{ message: string }> => {
  const response = await api.post(`/admin/tools/notification-resend/${dealerId}`);
  return response.data;
};

export const exportLeads = async (params: { status?: string; province?: string; start_date?: string; end_date?: string }): Promise<{ leads: ExtendedLead[]; count: number }> => {
  const response = await api.get('/admin/tools/lead-export', { params });
  return response.data;
};

export const changePassword = async (data: { current_password: string; new_password: string }): Promise<{ message: string }> => {
  const response = await api.post('/admin/tools/change-password', data);
  return response.data;
};

// ============================================
// AI ENDPOINTS
// ============================================

export const scoreLeadAI = async (leadId: number): Promise<AILeadScore> => {
  const response = await api.post('/ai/lead-score', null, { params: { lead_id: leadId } });
  return response.data;
};

export const getLeadScoreAI = async (leadId: number): Promise<{ ai_priority: string; ai_score: number; ai_reasoning: string; ai_scored_at: string }> => {
  const response = await api.get(`/ai/lead-score/${leadId}`);
  return response.data;
};

export const explainMatchAI = async (leadId: number): Promise<AIMatchExplanation> => {
  const response = await api.post('/ai/explain-match', null, { params: { lead_id: leadId } });
  return response.data;
};

export const draftEmailAI = async (data: { lead_id: number; purpose: string; dealer_id?: number }): Promise<AIEmailDraft> => {
  const response = await api.post('/ai/draft-email', data);
  return response.data;
};

export const enrichLeadAI = async (leadId: number): Promise<AIEnrichment> => {
  const response = await api.post('/ai/enrich-lead', null, { params: { lead_id: leadId } });
  return response.data;
};

export const predictConversionAI = async (leadId: number): Promise<AIConversionPrediction> => {
  const response = await api.post('/ai/predict-conversion', null, { params: { lead_id: leadId } });
  return response.data;
};

export const getAIInsights = async (): Promise<{ insights: AIInsight[] }> => {
  const response = await api.get('/ai/insights');
  return response.data;
};

export const getChurnRisks = async (): Promise<{ at_risk_leads: AIChurnRisk[] }> => {
  const response = await api.get('/ai/churn-risks');
  return response.data;
};

// ============================================
// DEALER PORTAL
// ============================================

export const getDealerActiveLeads = async (): Promise<{ leads: ExtendedLead[]; count: number }> => {
  const response = await api.get('/dealer/active-leads');
  return response.data;
};

export const getDealerHistory = async (): Promise<{ leads: ExtendedLead[]; count: number }> => {
  const response = await api.get('/dealer/history');
  return response.data;
};

export const submitDealerInteraction = async (data: { lead_id: number; message: string }): Promise<{ message: string }> => {
  const response = await api.post('/dealer/submit-interaction', data);
  return response.data;
};

export const submitDealerWinLost = async (data: { lead_id: number; status: string; value_of_order?: number; reason?: string }): Promise<{ message: string }> => {
  const response = await api.post('/dealer/submit-win', data);
  return response.data;
};

// ============================================
// EMAIL INTELLIGENCE
// ============================================

export const getEmailSyncConfig = async (): Promise<EmailSyncConfig | null> => {
  const response = await api.get('/email-intel/config', { timeout: 30000 });
  return response.data;
};

export const saveEmailSyncConfig = async (config: Partial<EmailSyncConfig> & { ms_client_secret?: string }): Promise<{ success: boolean }> => {
  const response = await api.post('/email-intel/config', config);
  return response.data;
};

export const getOAuthAuthorizeUrl = async (): Promise<{ auth_url: string }> => {
  const response = await api.get('/email-intel/oauth/authorize');
  return response.data;
};

export const completeOAuthCallback = async (code: string): Promise<{ success: boolean; email: string }> => {
  const response = await api.post('/email-intel/oauth/callback', { code });
  return response.data;
};

export const triggerEmailSync = async (): Promise<EmailSyncResult> => {
  const response = await api.post('/email-intel/sync', {}, { timeout: 120000 });
  return response.data;
};

export const getEmailSyncStatus = async (): Promise<EmailSyncStatus> => {
  const response = await api.get('/email-intel/status', { timeout: 30000 });
  return response.data;
};

export const getLeadEmails = async (leadId: number): Promise<EmailMessage[]> => {
  const response = await api.get(`/email-intel/lead/${leadId}/emails`, { timeout: 30000 });
  return response.data;
};

export const getLeadEmailContext = async (leadId: number): Promise<EmailLeadContext> => {
  const response = await api.get(`/email-intel/lead/${leadId}/context`, { timeout: 30000 });
  return response.data;
};

export const getClosureReviewQueue = async (): Promise<ClosureReview[]> => {
  const response = await api.get('/email-intel/review-queue', { timeout: 30000 });
  return response.data;
};

export const approveClosureReview = async (reviewId: number): Promise<{ success: boolean }> => {
  const response = await api.post(`/email-intel/review-queue/${reviewId}/approve`);
  return response.data;
};

export const dismissClosureReview = async (reviewId: number): Promise<{ success: boolean }> => {
  const response = await api.post(`/email-intel/review-queue/${reviewId}/dismiss`);
  return response.data;
};

export default api;
