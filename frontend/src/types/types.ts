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
  assigned_dealer_id?: number | null;
  dealer_name_assigned?: string | null;
  recommended_dealer_id?: number | null;
  recommended_dealer_name?: string | null;
  final_dealer_selection?: string | null;
  allocation_score?: number;
  distance_to_dealer_km?: number;
  dealer_ml_probability?: number;
  distance_review_required?: boolean;
  alternative_dealers?: AlternativeDealer[];
  created_at: string;
  updated_at?: string;
  latitude?: number;
  longitude?: number;
}

export interface AlternativeDealer {
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
  active_dealers: number;
}

export interface LassoDashboardStatus {
  sync_in_progress: boolean;
  sync_type?: string | null;
  started_at?: string | null;
  last_error?: string | null;
  last_successful_sync_at?: string | null;
  last_successful_sync_type?: string | null;
  last_full_successful_sync_at?: string | null;
  unassigned_count: number;
  active_count: number;
  history_count?: number;
  dealer_lead_reporting_synced_at?: string | null;
  history_synced_at?: string | null;
  dealer_performance_synced_at?: string | null;
  dealer_project_breakdown_synced_at?: string | null;
  dealer_status_synced_at?: string | null;
  fast_sync_interval_minutes: number;
  full_sync_interval_minutes: number;
  active_queue_max_age_days: number;
}

export interface DashboardUnassignedLead {
  lasso_lead_id: number;
  lead_id?: number | null;
  first_name?: string | null;
  last_name?: string | null;
  name: string;
  email?: string | null;
  city?: string | null;
  province?: string | null;
  location_text?: string | null;
  current_status?: string | null;
  record_date?: string | null;
  last_interaction?: string | null;
  synced_at: string;
}

export interface DashboardActiveLead {
  lasso_lead_id: number;
  lead_id?: number | null;
  dealer_id?: number | null;
  dealer_name?: string | null;
  first_name?: string | null;
  last_name?: string | null;
  name: string;
  email?: string | null;
  city?: string | null;
  province?: string | null;
  location_text?: string | null;
  current_status?: string | null;
  date_assigned?: string | null;
  last_interaction?: string | null;
  lead_details?: string | null;
  synced_at: string;
}

export interface CostTrackingModelBreakdown {
  model_used: string;
  total_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  tokens_used: number;
  spend_cad: number;
}

export interface CostTrackingPeriod {
  spend_cad: number;
  limit_cad: number;
  remaining_cad: number;
  total_calls: number;
  tokens_used: number;
  by_model: CostTrackingModelBreakdown[];
}

export interface CostTrackingSnapshot {
  daily: CostTrackingPeriod;
  monthly: CostTrackingPeriod;
  limits: {
    daily_spend_limit_cad: number;
    monthly_spend_limit_cad: number;
  };
}

export interface AIControlsSnapshot {
  agent_enabled: boolean;
  daily_spend_limit_cad: number;
  monthly_spend_limit_cad: number;
  role: string;
  can_manage: boolean;
}

export interface AISpendLimitUpdateRequest {
  daily_spend_limit_cad: number;
  monthly_spend_limit_cad: number;
}

export interface LeadsResponse {
  leads: Lead[];
  count: number;
  total: number;
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
  final_dealer_selection?: string;
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
// EMAIL INTELLIGENCE TYPES
// ============================================

export interface EmailSyncConfig {
  id?: number;
  configured?: boolean;
  is_connected?: boolean;
  ms_tenant_id: string;
  ms_client_id: string;
  ms_redirect_uri: string;
  shared_mailbox_email?: string | null;
  target_mailbox_type?: 'connected' | 'shared' | 'group';
  target_mailbox_email?: string | null;
  target_group_id?: string | null;
  sync_enabled: boolean;
  sync_interval_minutes: number;
  last_sync_at: string | null;
  user_email: string | null;
}

export interface EmailMessage {
  id: number;
  ms_message_id: string;
  subject: string;
  sender_email: string;
  sender_name: string;
  body_preview: string;
  body_text?: string;
  received_at: string;
  direction: 'inbound' | 'outbound';
  matched_lead_id: number | null;
  match_confidence: number | null;
  match_method: string | null;
  ai_summary: string | null;
  ai_sentiment: 'positive' | 'neutral' | 'negative' | null;
  ai_action_items: string[] | null;
  ai_ready_to_close: boolean;
  ai_urgency: 'high' | 'medium' | 'low' | null;
  ai_job_type: 'residential' | 'commercial' | 'replacement' | 'unknown' | null;
  ai_product: string | null;
  ai_window_count: string | null;
  ai_next_action: string | null;
  lead_first_name?: string | null;
  lead_last_name?: string | null;
  lead_status?: string | null;
}

export interface ClosureReview {
  id: number;
  lead_id: number;
  lead_name: string;
  lead_email: string;
  dealer_name: string;
  flagged_at: string;
  ai_reasoning: string;
  days_inactive: number;
  last_email_at: string | null;
  email_count: number;
  status: 'pending' | 'approved' | 'dismissed';
}

export interface ActiveMatchReviewItem {
  lead_id: number;
  lead_name: string;
  lead_email: string;
  lead_phone: string;
  lead_status: 'active' | 'follow_up';
  lead_source: string;
  landing_page?: string;
  landing_page_url?: string;
  assigned_dealer_id: number | null;
  assigned_dealer_name?: string | null;
  email_match_count: number;
  matched_email_count: number;
  weak_match_count: number;
  strong_match_count: number;
  max_match_confidence: number | null;
  match_methods: string;
  contact_emails: string;
  latest_email_id: number | null;
  latest_sender_email: string;
  latest_sender_name: string;
  latest_subject: string;
  latest_match_method: string;
  latest_match_confidence: number | null;
  latest_ai_summary: string;
  latest_ai_sentiment: 'positive' | 'neutral' | 'negative';
  latest_ai_ready_to_close: boolean;
  last_email_activity?: string | null;
  latest_email_at?: string | null;
  created_at: string;
  lead_age_days: number;
  days_since_last_email: number | null;
  active_duplicate_email_count: number;
  missing_dealer: boolean;
  needs_match_review: boolean;
  review_priority: 'high' | 'medium' | 'low';
  review_reason: string;
}

export interface NewLeadCandidate {
  id: number;
  received_at: string;
  sender_email: string;
  sender_name: string;
  subject: string;
  body_preview: string;
  candidate_reason: string;
  candidate_score: number;
  existing_sender_lead_count: number;
}

export interface EmailCandidateActionResult {
  success: boolean;
  action: 'created' | 'matched_existing';
  lead_id: number;
  matched_email_count: number;
}

export interface EmailLeadContext {
  context: string;
  timeline: Array<{
    date: string;
    direction?: string;
    subject?: string;
    summary?: string;
    sentiment: string;
  }>;
  key_insights: string[];
  recommended_action: string;
}

export interface EmailSyncStatus {
  sync_enabled: boolean;
  last_sync: string | null;
  total_emails: number;
  matched_emails: number;
  pending_reviews: number;
  sync_in_progress?: boolean;
  sync_started_at?: string | null;
  sync_finished_at?: string | null;
  sync_message?: string | null;
  last_sync_error?: string | null;
  last_sync_result?: EmailSyncResult | null;
  current_sync_counts?: {
    synced: number;
    matched: number;
    flagged_for_review: number;
  } | null;
}

export interface EmailSyncResult {
  synced?: number;
  matched?: number;
  flagged_for_review?: number;
  started?: boolean;
  sync_in_progress?: boolean;
  message?: string;
  result?: {
    synced: number;
    matched: number;
    flagged_for_review: number;
  } | null;
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
