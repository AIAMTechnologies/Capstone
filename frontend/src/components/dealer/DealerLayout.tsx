import React from 'react';
import { Outlet } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

const DealerLayout: React.FC = () => {
  const { logout } = useAuth();

  return (
    <div style={{ minHeight: '100vh', background: '#f5f6fa' }}>
      {/* Top nav */}
      <nav style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '0 24px', height: 56, background: '#1a1a2e', color: 'white',
      }}>
        <span style={{ fontSize: 16, fontWeight: 700 }}>Window Film Canada - Dealer Portal</span>
        <button
          onClick={logout}
          style={{
            padding: '6px 16px', background: '#c91414', color: 'white',
            border: 'none', borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: 'pointer',
          }}
        >
          Logout
        </button>
      </nav>

      {/* Content */}
      <div>
        <Outlet />
      </div>
    </div>
  );
};

export default DealerLayout;
