import React, { useState } from 'react';
import { createNewLead } from '../../services/api';

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

const INITIAL_FORM: AdminLeadFormState = {
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

interface InsertLeadFormProps {
  onLeadCreated?: () => void;
}

const InsertLeadForm: React.FC<InsertLeadFormProps> = ({ onLeadCreated }) => {
  const [form, setForm] = useState<AdminLeadFormState>(INITIAL_FORM);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleChange = (field: string, value: string) => {
    setForm((prev) => {
      if (field === 'country') {
        return { ...prev, country: value, province: '' };
      }
      return { ...prev, [field]: value };
    });
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!form.lead_source) {
      setError('This is a required field. Please select a lead source.');
      return;
    }

    const selectedProducts = [
      form.products_services_1,
      form.products_services_2,
      form.products_services_3
    ].filter(Boolean);

    if (new Set(selectedProducts).size !== selectedProducts.length) {
      setError('Please select different values for Products / Services 1-3.');
      return;
    }

    setSubmitting(true);

    try {
      const payload: Record<string, unknown> = {
        first_name: form.first_name,
        last_name: form.last_name,
        job_title: form.title,
        phone: form.primary_phone || form.cell_phone || form.work_phone,
        work_phone: form.work_phone,
        cell_phone: form.cell_phone,
        email: form.email,
        company_name: form.company,
        address: [form.address_line_1, form.address_line_2].filter(Boolean).join(', '),
        city: form.city,
        province: form.province,
        country: form.country,
        postal_code: form.postal_code,
        product_type: form.products_services_1,
        product_type_2: form.products_services_2,
        product_type_3: form.products_services_3,
        square_footage: form.square_footage,
        custom_pick_1: form.custom_pick_1,
        project_city: form.project_city,
        project_type: form.project_type,
        business_category: form.business_category,
        dealer_email: form.dealer_email,
        other_please_specify: form.other_please_specify,
        form_submit_date: form.date_yyyy_mm_dd || new Date().toISOString().slice(0, 10),
        lead_source: form.lead_source,
        opt_in: form.opt_in === 'Yes',
        landing_page: form.page_name,
        landing_page_url: form.url,
        landing_page_variant: form.variant,
        utm_source: form.utm_source,
        utm_medium: form.utm_medium,
        utm_campaign: form.utm_campaign,
        utm_content: form.utm_content,
        custom_pick_3: form.custom_pick_3
      };

      // Remove empty strings
      Object.keys(payload).forEach((key) => {
        if (payload[key] === '' || payload[key] === undefined) {
          delete payload[key];
        }
      });

      await createNewLead(payload);
      setForm(INITIAL_FORM);
      setSuccess('Lead created successfully.');
      onLeadCreated?.();
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Failed to create lead.';
      setError(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } finally {
      setSubmitting(false);
    }
  };

  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '10px 12px',
    border: '1px solid #ddd',
    borderRadius: 6,
    fontSize: 14,
    marginBottom: 10,
    boxSizing: 'border-box',
    outline: 'none',
  };

  const selectStyle: React.CSSProperties = {
    ...inputStyle,
    appearance: 'auto' as any,
    backgroundColor: 'white',
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && (
        <div style={{
          marginBottom: 12,
          padding: '10px 14px',
          borderRadius: 6,
          backgroundColor: '#fdecea',
          color: '#c0392b',
          fontSize: 14,
          border: '1px solid #f5b7b1'
        }}>
          {error}
        </div>
      )}
      {success && (
        <div style={{
          marginBottom: 12,
          padding: '10px 14px',
          borderRadius: 6,
          backgroundColor: '#d4edda',
          color: '#155724',
          fontSize: 14,
          border: '1px solid #c3e6cb'
        }}>
          {success}
        </div>
      )}

      {/* Row 1: Contact + Address + Products */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 8 }}>
        <div>
          <input style={inputStyle} placeholder="First Name" value={form.first_name} onChange={(e) => handleChange('first_name', e.target.value)} />
          <input style={inputStyle} placeholder="Last Name" value={form.last_name} onChange={(e) => handleChange('last_name', e.target.value)} />
          <input style={inputStyle} placeholder="Title" value={form.title} onChange={(e) => handleChange('title', e.target.value)} />
          <input style={inputStyle} placeholder="Primary Phone" value={form.primary_phone} onChange={(e) => handleChange('primary_phone', e.target.value)} />
          <input style={inputStyle} placeholder="Work Phone" value={form.work_phone} onChange={(e) => handleChange('work_phone', e.target.value)} />
          <input style={inputStyle} placeholder="Cell Phone" value={form.cell_phone} onChange={(e) => handleChange('cell_phone', e.target.value)} />
          <input style={inputStyle} type="email" placeholder="Email" value={form.email} onChange={(e) => handleChange('email', e.target.value)} />
        </div>

        <div>
          <input style={inputStyle} placeholder="Company" value={form.company} onChange={(e) => handleChange('company', e.target.value)} />
          <input style={inputStyle} placeholder="Address Line 1" value={form.address_line_1} onChange={(e) => handleChange('address_line_1', e.target.value)} />
          <input style={inputStyle} placeholder="Address Line 2" value={form.address_line_2} onChange={(e) => handleChange('address_line_2', e.target.value)} />
          <input style={inputStyle} placeholder="City" value={form.city} onChange={(e) => handleChange('city', e.target.value)} />
          <select style={selectStyle} value={form.country} onChange={(e) => handleChange('country', e.target.value)}>
            <option value="">Country</option>
            {COUNTRY_OPTIONS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <select style={selectStyle} value={form.province} onChange={(e) => handleChange('province', e.target.value)}>
            <option value="">Province / State</option>
            {(COUNTRY_SUBDIVISIONS[form.country] || []).map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <input style={inputStyle} placeholder="Postal Code" value={form.postal_code} onChange={(e) => handleChange('postal_code', e.target.value)} />
        </div>

        <div>
          <select style={selectStyle} value={form.products_services_1} onChange={(e) => handleChange('products_services_1', e.target.value)}>
            <option value="">Products / Services 1</option>
            {PRODUCT_SERVICE_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          <select style={selectStyle} value={form.products_services_2} onChange={(e) => handleChange('products_services_2', e.target.value)}>
            <option value="">Products / Services 2</option>
            {PRODUCT_SERVICE_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          <select style={selectStyle} value={form.products_services_3} onChange={(e) => handleChange('products_services_3', e.target.value)}>
            <option value="">Products / Services 3</option>
            {PRODUCT_SERVICE_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          <select style={selectStyle} value={form.square_footage} onChange={(e) => handleChange('square_footage', e.target.value)}>
            <option value="">Square Footage</option>
            {SQUARE_FOOTAGE_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <input style={inputStyle} placeholder="Custom Pick 1" value={form.custom_pick_1} onChange={(e) => handleChange('custom_pick_1', e.target.value)} />
          <input style={inputStyle} placeholder="Project City" value={form.project_city} onChange={(e) => handleChange('project_city', e.target.value)} />
          <select style={selectStyle} value={form.project_type} onChange={(e) => handleChange('project_type', e.target.value)}>
            <option value="">Project Type</option>
            {PROJECT_TYPE_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
      </div>

      {/* Row 2: Business, Dealer, Source */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 8 }}>
        <div>
          <input style={inputStyle} placeholder="Business Category" value={form.business_category} onChange={(e) => handleChange('business_category', e.target.value)} />
          <input style={inputStyle} type="email" placeholder="Dealer Email" value={form.dealer_email} onChange={(e) => handleChange('dealer_email', e.target.value)} />
        </div>
        <div>
          <input style={inputStyle} placeholder="Other Please Specify" value={form.other_please_specify} onChange={(e) => handleChange('other_please_specify', e.target.value)} />
          <input style={inputStyle} type="date" value={form.date_yyyy_mm_dd} onChange={(e) => handleChange('date_yyyy_mm_dd', e.target.value)} />
        </div>
        <div>
          <select style={selectStyle} value={form.lead_source} onChange={(e) => handleChange('lead_source', e.target.value)}>
            <option value="">Lead Source *</option>
            {LEAD_SOURCE_OPTIONS.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
          <select style={selectStyle} value={form.opt_in} onChange={(e) => handleChange('opt_in', e.target.value)}>
            <option value="">Opt-In</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>
      </div>

      {/* Row 3: UTM / Tracking */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 8 }}>
        <input style={inputStyle} placeholder="Page Name" value={form.page_name} onChange={(e) => handleChange('page_name', e.target.value)} />
        <input style={inputStyle} placeholder="URL" value={form.url} onChange={(e) => handleChange('url', e.target.value)} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 8 }}>
        <input style={inputStyle} placeholder="Variant" value={form.variant} onChange={(e) => handleChange('variant', e.target.value)} />
        <input style={inputStyle} placeholder="UTM Source" value={form.utm_source} onChange={(e) => handleChange('utm_source', e.target.value)} />
        <input style={inputStyle} placeholder="UTM Medium" value={form.utm_medium} onChange={(e) => handleChange('utm_medium', e.target.value)} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 8 }}>
        <input style={inputStyle} placeholder="UTM Campaign" value={form.utm_campaign} onChange={(e) => handleChange('utm_campaign', e.target.value)} />
        <input style={inputStyle} placeholder="UTM Content" value={form.utm_content} onChange={(e) => handleChange('utm_content', e.target.value)} />
        <input style={inputStyle} placeholder="Custom Pick 3" value={form.custom_pick_3} onChange={(e) => handleChange('custom_pick_3', e.target.value)} />
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
        <button
          type="submit"
          disabled={submitting}
          style={{
            padding: '10px 24px',
            borderRadius: 6,
            border: 'none',
            cursor: submitting ? 'not-allowed' : 'pointer',
            fontSize: 14,
            fontWeight: 600,
            background: '#c91414',
            color: 'white',
            opacity: submitting ? 0.6 : 1,
          }}
        >
          {submitting ? 'Submitting...' : 'Insert New Lead'}
        </button>
      </div>
    </form>
  );
};

export default InsertLeadForm;
