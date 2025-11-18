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
  HistoricalDataResponse
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
): Promise<{ message: string; lead_id: number; installer_id: number | null }> => {
  const response = await api.patch(`/admin/leads/${leadId}/installer-override`, null, {
    params: { installer_id: installerId }
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

export default api;