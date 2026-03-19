const normalizeApiUrl = (rawUrl: string | undefined | null): string => {
  const trimmed = (rawUrl ?? '').trim();
  if (!trimmed) {
    return '';
  }

  return trimmed.endsWith('/') ? trimmed.replace(/\/+$/, '') : trimmed;
};

const rawApiUrl = normalizeApiUrl(import.meta.env.VITE_API_URL) || 'http://localhost:8000/api';
const rawGoogleMapsKey = (import.meta.env.VITE_GOOGLE_MAPS_API_KEY ?? '').trim();
const isPlaceholderGoogleKey = /YOUR_FRONTEND_GOOGLE_KEY/i.test(rawGoogleMapsKey);
const googleMapsApiKey = rawGoogleMapsKey && !isPlaceholderGoogleKey ? rawGoogleMapsKey : '';

export const env = {
  apiUrl: rawApiUrl,
  rawGoogleMapsKey,
  googleMapsApiKey,
  isGoogleMapsKeyPlaceholder: isPlaceholderGoogleKey,
};
