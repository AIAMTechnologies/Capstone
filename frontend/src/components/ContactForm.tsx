import React, { useState, useRef, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { Autocomplete, useLoadScript } from '@react-google-maps/api';
import { MapPin, User, Mail, Phone, Home, MessageSquare, Send, CheckCircle } from 'lucide-react';
import { submitLead, getPublicGoogleMapsApiKey } from '../services/api';
import type { LeadFormData } from '../types';
import { env } from '../config/env';

const libraries: ("places")[] = ['places'];

interface ContactFormContentProps {
  googleMapsApiKey: string;
}

const ContactFormContent: React.FC<ContactFormContentProps> = ({ googleMapsApiKey }) => {
  const [autocomplete, setAutocomplete] = useState<google.maps.places.Autocomplete | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [addressError, setAddressError] = useState('');
  const addressInputRef = useRef<HTMLInputElement>(null);

  const { register, handleSubmit, formState: { errors }, setValue, reset } = useForm<LeadFormData>();

  // Load Google Maps script
  const { isLoaded, loadError } = useLoadScript({
    googleMapsApiKey,
    libraries,
  });

  const onLoad = (autocompleteInstance: google.maps.places.Autocomplete) => {
    console.log('Autocomplete loaded');
    setAutocomplete(autocompleteInstance);
  };

  const onPlaceChanged = () => {
    console.log('Place changed');
    if (autocomplete !== null) {
      const place = autocomplete.getPlace();
      console.log('Place:', place);
      
      if (place.address_components) {
        let streetNumber = '';
        let route = '';
        let city = '';
        let province = '';
        let postalCode = '';

        place.address_components.forEach((component) => {
          const types = component.types;
          
          if (types.includes('street_number')) {
            streetNumber = component.long_name;
          }
          if (types.includes('route')) {
            route = component.long_name;
          }
          if (types.includes('locality')) {
            city = component.long_name;
          }
          if (types.includes('administrative_area_level_1')) {
            province = component.short_name;
          }
          if (types.includes('postal_code')) {
            postalCode = component.long_name;
          }
        });

        const fullAddress = `${streetNumber} ${route}`.trim();
        
        // Update the input field with just the street address
        if (addressInputRef.current) {
          addressInputRef.current.value = fullAddress;
        }
        
        setValue('address', fullAddress);
        setValue('city', city);
        setValue('province', province as any);
        setValue('postal_code', postalCode);
        setAddressError('');
      }
    }
  };

  const onSubmit = async (data: LeadFormData) => {
    // Get address value from the input element
    const addressValue = addressInputRef.current?.value || '';
    
    // Validate address manually since it's not registered with react-hook-form
    if (!addressValue || addressValue.length < 5) {
      setAddressError('Address must be at least 5 characters');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      // Add address to data
      const submitData = { ...data, address: addressValue };
      await submitLead(submitData);
      setSuccess(true);
      reset();
      if (addressInputRef.current) {
        addressInputRef.current.value = '';
      }
      
      // Reset success message after 5 seconds
      setTimeout(() => setSuccess(false), 5000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit form. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (loadError) {
    return <div className="alert alert-error">Error loading Google Maps</div>;
  }

  if (!isLoaded) {
    return <div className="spinner"></div>;
  }

  return (
    <div className="card" style={{ maxWidth: '800px', margin: '40px auto' }}>
      <div className="card-header">
        <h2>Request a Quote</h2>
      </div>
      
      {success && (
        <div className="alert alert-success">
          <CheckCircle size={24} />
          <div>
            <strong>Success!</strong> Your request has been submitted. We'll contact you soon.
          </div>
        </div>
      )}

      {error && (
        <div className="alert alert-error">
          <strong>Error:</strong> {error}
        </div>
      )}

      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="grid grid-2">
          <div className="form-group">
            <label className="form-label">
              <User size={16} /> Full Name *
            </label>
            <input
              type="text"
              className="form-input"
              {...register('name', { 
                required: 'Name is required',
                minLength: { value: 2, message: 'Name must be at least 2 characters' }
              })}
              placeholder="John Doe"
            />
            {errors.name && <p className="form-error">{errors.name.message}</p>}
          </div>

          <div className="form-group">
            <label className="form-label">
              <Mail size={16} /> Email Address *
            </label>
            <input
              type="email"
              className="form-input"
              {...register('email', { 
                required: 'Email is required',
                pattern: {
                  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                  message: 'Invalid email address'
                }
              })}
              placeholder="john@example.com"
            />
            {errors.email && <p className="form-error">{errors.email.message}</p>}
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">
            <Phone size={16} /> Phone Number *
          </label>
          <input
            type="tel"
            className="form-input"
            {...register('phone', { 
              required: 'Phone number is required',
              minLength: { value: 10, message: 'Phone number must be at least 10 digits' }
            })}
            placeholder="(123) 456-7890"
          />
          {errors.phone && <p className="form-error">{errors.phone.message}</p>}
        </div>

        <div className="form-group">
          <label className="form-label">
            <MapPin size={16} /> Street Address *
          </label>
          <Autocomplete
            onLoad={onLoad}
            onPlaceChanged={onPlaceChanged}
            options={{
              componentRestrictions: { country: 'ca' },
              types: ['address'],
              fields: ['address_components', 'formatted_address']
            }}
          >
            <input
              ref={addressInputRef}
              type="text"
              className="form-input"
              placeholder="Start typing your address..."
              autoComplete="new-password"
              name="street_address"
              onChange={(e) => {
                console.log('Input value:', e.target.value);
                if (e.target.value.length >= 5) {
                  setAddressError('');
                }
              }}
            />
          </Autocomplete>
          {addressError && <p className="form-error">{addressError}</p>}
          <p className="form-help">Start typing and select your address from the suggestions</p>
        </div>

        <div className="grid grid-3">
          <div className="form-group">
            <label className="form-label">
              <Home size={16} /> City *
            </label>
            <input
              type="text"
              className="form-input"
              {...register('city', { 
                required: 'City is required',
                minLength: { value: 2, message: 'City must be at least 2 characters' }
              })}
              placeholder="Toronto"
            />
            {errors.city && <p className="form-error">{errors.city.message}</p>}
          </div>

          <div className="form-group">
            <label className="form-label">Province *</label>
            <select
              className="form-select"
              {...register('province', { required: 'Province is required' })}
            >
              <option value="">Select Province</option>
              <option value="AB">Alberta</option>
              <option value="BC">British Columbia</option>
              <option value="MB">Manitoba</option>
              <option value="NB">New Brunswick</option>
              <option value="NL">Newfoundland and Labrador</option>
              <option value="NS">Nova Scotia</option>
              <option value="NT">Northwest Territories</option>
              <option value="NU">Nunavut</option>
              <option value="ON">Ontario</option>
              <option value="PE">Prince Edward Island</option>
              <option value="QC">Quebec</option>
              <option value="SK">Saskatchewan</option>
              <option value="YT">Yukon</option>
            </select>
            {errors.province && <p className="form-error">{errors.province.message}</p>}
          </div>

          <div className="form-group">
            <label className="form-label">Postal Code</label>
            <input
              type="text"
              className="form-input"
              {...register('postal_code')}
              placeholder="A1A 1A1"
            />
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Job Type *</label>
          <select
            className="form-select"
            {...register('job_type', { required: 'Job type is required' })}
          >
            <option value="">Select Job Type</option>
            <option value="residential">Residential</option>
            <option value="commercial">Commercial</option>
          </select>
          {errors.job_type && <p className="form-error">{errors.job_type.message}</p>}
        </div>

        <div className="form-group">
          <label className="form-label">
            <MessageSquare size={16} /> Additional Comments
          </label>
          <textarea
            className="form-textarea"
            {...register('comments')}
            placeholder="Tell us about your project..."
          />
        </div>

        <button 
          type="submit" 
          className="btn btn-primary" 
          disabled={loading}
          style={{ width: '100%' }}
        >
          {loading ? (
            <>Processing...</>
          ) : (
            <>
              <Send size={20} />
              Submit Request
            </>
          )}
        </button>
      </form>
    </div>
  );
};

const ContactForm: React.FC = () => {
  const compileTimeKey = env.googleMapsApiKey;
  const [resolvedKey, setResolvedKey] = useState<string | null>(compileTimeKey || null);
  const [fetchingRuntimeKey, setFetchingRuntimeKey] = useState(!compileTimeKey);
  const [keyError, setKeyError] = useState<string | null>(null);

  useEffect(() => {
    if (compileTimeKey) {
      setResolvedKey(compileTimeKey);
      setFetchingRuntimeKey(false);
      setKeyError(null);
      return;
    }

    let active = true;
    setFetchingRuntimeKey(true);
    setKeyError(null);

    (async () => {
      try {
        const runtimeKey = await getPublicGoogleMapsApiKey();
        if (!active) {
          return;
        }

        if (runtimeKey) {
          setResolvedKey(runtimeKey);
          setKeyError(null);
        } else {
          setKeyError('Google Maps is temporarily unavailable because the API key is missing from the server configuration.');
        }
      } catch (error) {
        if (!active) {
          return;
        }
        setKeyError('Unable to load the Google Maps API key from the server. Please try again or verify backend/.env.');
      } finally {
        if (active) {
          setFetchingRuntimeKey(false);
        }
      }
    })();

    return () => {
      active = false;
    };
  }, [compileTimeKey]);

  if (fetchingRuntimeKey) {
    return (
      <div className="card" style={{ maxWidth: '800px', margin: '40px auto' }}>
        <div className="form-group" style={{ textAlign: 'center' }}>
          <div className="spinner" style={{ margin: '0 auto 1rem' }} />
          <p>Loading Google Maps configuration…</p>
        </div>
      </div>
    );
  }

  if (!resolvedKey) {
    return (
      <div className="card" style={{ maxWidth: '800px', margin: '40px auto' }}>
        <div className="alert alert-error">
          {keyError || 'Google Maps is temporarily unavailable because the API key is missing.'}
          <div style={{ marginTop: '0.5rem' }}>
            Ensure <code>frontend/.env.local</code> contains a valid <code>VITE_GOOGLE_MAPS_API_KEY</code> and restart the Vite dev server.
            {env.isGoogleMapsKeyPlaceholder && (
              <>
                {' '}
                The placeholder value <code>YOUR_FRONTEND_GOOGLE_KEY</code> is ignored to prevent invalid key errors.
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  return <ContactFormContent googleMapsApiKey={resolvedKey} />;
};

export default ContactForm;