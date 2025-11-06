// Type definitions for the application

export interface Lead {
  id: number;
  name: string;
  email: string;
  phone: string;
  address: string;
  city: string;
  province: string;
  postal_code?: string;
  job_type: 'residential' | 'commercial';
  comments?: string;
  status: 'active' | 'converted' | 'dead';
  assigned_installer_id?: number;
  assigned_installer_name?: string;
  installer_name?: string;
  installer_city?: string;
  allocation_score?: number;
  distance_to_installer_km?: number;
  created_at: string;
  updated_at?: string;
  latitude?: number;
  longitude?: number;
}

export interface LeadFormData {
  name: string;
  email: string;
  phone: string;
  address: string;
  city: string;
  province: string;
  postal_code?: string;
  job_type: 'residential' | 'commercial';
  comments?: string;
}

export interface AdminUser {
  id: number;
  username: string;
  email: string;
  last_name?: string;
  role: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: AdminUser;
}

export interface DashboardStats {
  total_leads: number;
  pending_leads: number;
  assigned_leads: number;
  completed_leads: number;
  conversion_rate: number;
  avg_allocation_score: number;
  active_installers: number;
}

export interface LeadsResponse {
  leads: Lead[];
  count: number;
  total: number;
}

export interface Installer {
  id: number;
  name: string;
  email: string;
  phone: string;
  city: string;
  province: string;
  is_active: boolean;
  total_leads?: number;
  converted_leads?: number;
  active_leads?: number;
}

export type LeadStatus = 'active' | 'converted' | 'dead';

export type Province = 
  | 'AB' | 'BC' | 'MB' | 'NB' | 'NL' | 'NS' 
  | 'NT' | 'NU' | 'ON' | 'PE' | 'QC' | 'SK' | 'YT';

export interface HistoricalData {
  id: number;
  submit_date?: string;
  first_name?: string;
  last_name?: string;
  company_name?: string;
  address1?: string;
  city?: string;
  province?: string;
  postal?: string;
  dealer_name?: string;
  project_type?: string;
  product_type?: string;
  square_footage?: number;
  current_status?: string;
  job_won_date?: string;
  value_of_order?: number;
  job_lost_date?: string;
  reason?: string;
  created_at: string;
  updated_at?: string;
}

export interface HistoricalDataResponse {
  data: HistoricalData[];
  count: number;
  total: number;
}