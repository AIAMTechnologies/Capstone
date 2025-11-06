import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { LogOut, Home, LayoutDashboard } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const Header = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/admin/login');
  };

  return (
    <header style={styles.header}>
      <div style={styles.container}>
        <Link to="/" style={styles.logo}>
          <img 
            src="/logo.png" 
            alt="Window Film Canada" 
            style={styles.logoImage}
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'block';
            }}
          />
          <span style={styles.logoText}>WFC</span>
        </Link>

        <nav style={styles.nav}>
          {!isAuthenticated ? (
            <>
              <Link to="/" style={styles.navLink}>
                <Home size={20} />
                <span>Home</span>
              </Link>
              <Link to="/admin/login" style={styles.navLink}>
                <LayoutDashboard size={20} />
                <span>Admin Login</span>
              </Link>
            </>
          ) : (
            <>
              <Link to="/admin/dashboard" style={styles.navLink}>
                <LayoutDashboard size={20} />
                <span>Dashboard</span>
              </Link>
              <div style={styles.userInfo}>
                <span style={styles.username}>{user?.username}</span>
                <button onClick={handleLogout} style={styles.logoutBtn}>
                  <LogOut size={20} />
                  <span>Logout</span>
                </button>
              </div>
            </>
          )}
        </nav>
      </div>
    </header>
  );
};

const styles = {
  header: {
    backgroundColor: '#c91414',
    color: 'white',
    padding: '16px 0',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    position: 'sticky',
    top: 0,
    zIndex: 1000,
  },
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    color: 'white',
    textDecoration: 'none',
    fontSize: '24px',
    fontWeight: 'bold',
  },
  logoImage: {
    height: '40px',
    width: 'auto',
  },
  logoText: {
    display: 'block',
  },
  nav: {
    display: 'flex',
    alignItems: 'center',
    gap: '24px',
  },
  navLink: {
    color: 'white',
    textDecoration: 'none',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 16px',
    borderRadius: '8px',
    transition: 'background-color 0.3s',
    fontSize: '16px',
  },
  userInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  },
  username: {
    fontSize: '16px',
    fontWeight: '600',
  },
  logoutBtn: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    color: 'white',
    border: 'none',
    padding: '8px 16px',
    borderRadius: '8px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '16px',
    transition: 'background-color 0.3s',
  },
};

export default Header;