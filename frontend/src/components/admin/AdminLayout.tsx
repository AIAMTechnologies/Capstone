import React from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { LayoutDashboard, Clock, BookOpen, BarChart3, Wrench, LogOut } from 'lucide-react';

const NAV_ITEMS = [
  { to: '/admin/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/admin/history', label: 'History', icon: Clock },
  { to: '/admin/resources', label: 'Resources', icon: BookOpen },
  { to: '/admin/reports', label: 'Reports', icon: BarChart3 },
  { to: '/admin/tools', label: 'Tools', icon: Wrench },
];

const AdminLayout: React.FC = () => {
  const { logout, user } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/admin/login');
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <aside style={{
        width: 220,
        background: '#1a1a2e',
        color: '#fff',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}>
        <div style={{ padding: '20px 16px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#e74c3c', margin: 0 }}>WFC Admin</h2>
          {user && <p style={{ fontSize: 12, color: '#aaa', marginTop: 4 }}>{user.username}</p>}
        </div>
        <nav style={{ flex: 1, padding: '12px 0' }}>
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '10px 16px',
                color: isActive ? '#fff' : '#aaa',
                background: isActive ? 'rgba(231,76,60,0.2)' : 'transparent',
                borderLeft: isActive ? '3px solid #e74c3c' : '3px solid transparent',
                textDecoration: 'none',
                fontSize: 14,
                transition: 'all 0.2s',
              })}
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>
        <button
          onClick={handleLogout}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '12px 16px',
            color: '#aaa',
            background: 'none',
            border: 'none',
            borderTop: '1px solid rgba(255,255,255,0.1)',
            cursor: 'pointer',
            fontSize: 14,
            width: '100%',
            textAlign: 'left',
          }}
        >
          <LogOut size={18} />
          Logout
        </button>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, background: '#f5f6fa', overflow: 'auto' }}>
        <Outlet />
      </main>
    </div>
  );
};

export default AdminLayout;
