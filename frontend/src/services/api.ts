import axios, { AxiosError } from 'axios';
import type { 
  LeadFormData, 
  LoginResponse, 
  DashboardStats, 
  LeadsResponse, 
  Lead,
  LeadStatus,
  Installer
} from '../types';

const normalizeUrl = (url: string) => url.replace(/\/+$/, '');

const LOCAL_DEV_API = 'http://localhost:8000/api';

const getApiBaseUrl = (): string => {
  const rawApiUrl = import.meta.env.VITE_API_URL?.trim();

  if (rawApiUrl) {
    return normalizeUrl(rawApiUrl);
  }

  if (typeof window !== 'undefined') {
    const { origin, hostname } = window.location;

    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      console.warn(
        'VITE_API_URL is not defined. Using the local backend at http://localhost:8000/api; set VITE_API_URL to silence this warning.'
      );
      return normalizeUrl(LOCAL_DEV_API);
    }

    if (origin) {
      console.warn(
        'VITE_API_URL is not defined. Falling back to the current origin for API requests; set VITE_API_URL to silence this warning.'
      );
      return `${normalizeUrl(origin)}/api`;
    }
  }

  console.warn(
    'VITE_API_URL is not defined. Falling back to the relative /api path; set VITE_API_URL to silence this warning.'
  );
  return '/api';
};

const API_BASE_URL = getApiBaseUrl();

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

export const getInstallers = async (): Promise<{ installers: Installer[]; count: number }> => {
  const response = await api.get('/admin/installers');
  return response.data;
};

export default api;
