export const getApiErrorMessage = (error: unknown, fallback: string): string => {
  const detail = (error as any)?.response?.data?.detail;

  if (!detail) {
    return (error as any)?.message || fallback;
  }

  if (typeof detail === 'string') {
    return detail;
  }

  if (Array.isArray(detail)) {
    return detail.map((item: any) => item?.msg || JSON.stringify(item)).join('; ');
  }

  if (typeof detail === 'object') {
    return detail.msg || JSON.stringify(detail);
  }

  return fallback;
};
