import React from 'react';

const cardStyle: React.CSSProperties = {
  background: 'white',
  borderRadius: 8,
  padding: 24,
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  maxWidth: 760,
};

const Resources: React.FC = () => {
  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: '#1a1a2e', marginBottom: 20 }}>Resources</h1>
      <div style={cardStyle}>
        <h2 style={{ fontSize: 18, fontWeight: 600, marginTop: 0, marginBottom: 12 }}>Resources Retired</h2>
        <p style={{ color: '#4b5563', lineHeight: 1.7, margin: 0 }}>
          This internal resource library has been retired as part of the Lasso-only rewrite. The admin portal now focuses on live Lasso-backed operational data instead of locally managed content pages.
        </p>
      </div>
    </div>
  );
};

export default Resources;
