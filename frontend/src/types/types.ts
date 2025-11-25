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
  status: 'active' | 'converted' | 'dead'|'follow_up' ;
  assigned_installer_id?: number | null;
  assigned_installer_name?: string | null;
  installer_name?: string | null;
  final_installer_selection?: string | null;
  installer_city?: string | null;
  allocation_score?: number;
  distance_to_installer_km?: number;
  installer_ml_probability?: number;
  distance_review_required?: boolean;
  installer_override_id?: number | null;
  alternative_installers?: AlternativeInstaller[];
  created_at: string;
  updated_at?: string;
  latitude?: number;
  longitude?: number;
}

export interface AlternativeInstaller {
  id: number;
  name: string;
  city: string;
  province: string;
  distance_km: number;
  allocation_score: number;
  active_leads: number;
  converted_leads?: number;
  ml_probability?: number;
  distance_review_required?: boolean;
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

export type LeadStatus = 'active' | 'converted' | 'dead'| 'follow_up';

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
  final_installer_selection?: string;
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