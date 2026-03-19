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

// ============================================
// DEALER TYPES
// ============================================

export interface Dealer {
  id: number;
  name: string;
  city: string;
  province: string;
  email: string;
  notification_email?: string;
  is_active: boolean;
}

// ============================================
// EXTENDED LEAD (includes all 35+ fields)
// ============================================

export interface ExtendedLead extends Lead {
  first_name?: string;
  last_name?: string;
  job_title?: string;
  company_name?: string;
  product_type?: string;
  product_type_2?: string;
  product_type_3?: string;
  square_footage?: number;
  custom_pick_1?: string;
  project_city?: string;
  project_type?: string;
  business_category?: string;
  dealer_email?: string;
  form_submit_date?: string;
  lead_source?: string;
  opt_in?: boolean;
  landing_page?: string;
  landing_page_url?: string;
  landing_page_variant?: string;
  utm_source?: string;
  utm_medium?: string;
  utm_campaign?: string;
  utm_content?: string;
  utm_term?: string;
  custom_pick_3?: string;
  value_of_order?: number;
  assigned_dealer_id?: number | null;
  dealer_name_assigned?: string | null;
  // AI fields
  ai_priority?: string;
  ai_score?: number;
  ai_reasoning?: string;
  ai_scored_at?: string;
  ai_match_explanation?: string;
  ai_conversion_likelihood?: number;
  ai_conversion_explanation?: string;
}

// ============================================
// LEAD LOG & ASSIGNMENT TYPES
// ============================================

export interface LeadLog {
  id: number;
  lead_id: number;
  log_type: string;
  message: string;
  created_by?: string;
  dealer_id?: number;
  dealer_name?: string;
  created_at: string;
}

export interface LeadAssignment {
  id: number;
  lead_id: number;
  dealer_id: number;
  dealer_name?: string;
  assigned_at: string;
  responded_at?: string;
  response_time_hours?: number;
  status: string;
}

// ============================================
// RESOURCE TYPES
// ============================================

export interface ResourcePage {
  id: number;
  parent_id: number | null;
  title: string;
  slug: string;
  content: string;
  sort_order: number;
  is_published: boolean;
  created_at: string;
  updated_at: string;
}

// ============================================
// NOTIFICATION TYPES
// ============================================

export interface Notification {
  id: number;
  dealer_id: number;
  notification_type: string;
  sent_at: string;
  responded: boolean;
  responded_at?: string;
}

export interface DealerNotificationStatus {
  id: number;
  name: string;
  email: string;
  total_notifications: number;
  responded: number;
  last_sent?: string;
  last_responded?: string;
}

// ============================================
// AI TYPES
// ============================================

export interface AILeadScore {
  priority: 'Hot' | 'Warm' | 'Cold';
  score: number;
  reasoning: string;
  suggested_actions: string[];
}

export interface AIMatchExplanation {
  explanation: string;
  confidence: string;
  considerations: string[];
}

export interface AIInsight {
  type: 'warning' | 'trend' | 'alert';
  title: string;
  body: string;
  action: string;
}

export interface AIEmailDraft {
  subject: string;
  body: string;
  tone: string;
}

export interface AIEnrichment {
  inferred_business_category: string;
  estimated_project_size: string;
  suggested_products: string[];
  confidence: number;
}

export interface AIConversionPrediction {
  likelihood: number;
  label: string;
  explanation: string;
  risk_factors: string[];
}

export interface AIChurnRisk {
  lead_id: number;
  name?: string;
  risk_level: string;
  days_inactive: number;
  reason: string;
  recommended_action: string;
  dealer_name?: string;
}

// ============================================
// REPORT TYPES
// ============================================

// Tab 1: Overall Summary row
export interface LeadReportSummary {
  timeframe: string;
  total_leads: number;
  converted: number;
  dead: number;
  active: number;
  converted_pct: number;
  dead_pct: number;
  active_pct: number;
}

export interface GraphDataPoint {
  period: string;
  total: number;
  converted: number;
  dead: number;
}

// Tab 2: Dealer Performance (Report 1)
export interface DealerPerformanceData {
  dealer_id: number;
  dealer_name: string;
  active_leads: number;
  converted: number;
  dead: number;
  total_leads: number;
  avg_response_hours: number | null;
  avg_response_str: string;
}

// Tab 3: Project Size Breakdown (Report 2)
export interface DealerProjectData {
  dealer_id: number;
  dealer_name: string;
  sqft_1_499: number;
  sqft_500_999: number;
  sqft_1000_3499: number;
  sqft_3500_7499: number;
  sqft_7500_19999: number;
  sqft_20000_plus: number;
  total_leads: number;
}

// Tab 4: Lead Status & Conversion Scores (Report 3)
export interface LeadStatusReport {
  dealer_id: number;
  dealer_name: string;
  reviewing_undecided: number;
  building_budget: number;
  converted_total_value: number;
  total_leads: number;
  converted_count: number;
  lead_score_pct: number;
}

// Legacy compat
export interface ResponseTimeData {
  dealer_name: string;
  dealer_id: number;
  total_assignments: number;
  avg_response_hours: number | null;
  min_response_hours: number | null;
  max_response_hours: number | null;
}