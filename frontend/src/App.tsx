import React, { lazy, Suspense } from 'react';
import { createBrowserRouter, RouterProvider, Navigate, Outlet } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import Header from './components/Header';
import ProtectedRoute from './components/ProtectedRoute';
import Home from './pages/Home';
import AdminLogin from './pages/AdminLogin';
import AdminLayout from './components/admin/AdminLayout';
import './styles/App.css';

// Lazy-load all admin pages so /admin/dashboard doesn't pull in
// Email Intel, Reports, History, etc. up front.
const Dashboard = lazy(() => import('./pages/admin/Dashboard'));
const History = lazy(() => import('./pages/admin/History'));
const Resources = lazy(() => import('./pages/admin/Resources'));
const Reports = lazy(() => import('./pages/admin/Reports'));
const Tools = lazy(() => import('./pages/admin/Tools'));
const EmailIntel = lazy(() => import('./pages/admin/EmailIntel'));
const EmailOps = lazy(() => import('./pages/admin/EmailOps'));

const AdminFallback = () => (
  <div style={{ padding: '40px 20px', color: '#6b7280' }}>Loading...</div>
);

const AppShell = () => (
  <div className="app">
    <Header />
    <main>
      <Outlet />
    </main>
  </div>
);

const AdminShell = () => (
  <ProtectedRoute>
    <AdminLayout />
  </ProtectedRoute>
);

const router = createBrowserRouter(
  [
    {
      element: <AppShell />,
      children: [
        { path: '/', element: <Home /> },
        { path: '/admin/login', element: <AdminLogin /> },
        { path: '*', element: <Navigate to="/" replace /> },
      ],
    },
    {
      path: '/admin',
      element: <AdminShell />,
      children: [
        { index: true, element: <Navigate to="/admin/dashboard" replace /> },
        { path: 'dashboard', element: <Suspense fallback={<AdminFallback />}><Dashboard /></Suspense> },
        { path: 'history', element: <Suspense fallback={<AdminFallback />}><History /></Suspense> },
        { path: 'resources', element: <Suspense fallback={<AdminFallback />}><Resources /></Suspense> },
        { path: 'reports', element: <Suspense fallback={<AdminFallback />}><Reports /></Suspense> },
        { path: 'tools', element: <Suspense fallback={<AdminFallback />}><Tools /></Suspense> },
        { path: 'email-intel', element: <Suspense fallback={<AdminFallback />}><EmailIntel /></Suspense> },
        { path: 'email-ops', element: <Suspense fallback={<AdminFallback />}><EmailOps /></Suspense> },
      ],
    },
  ]
);

function App() {
  return (
    <AuthProvider>
      <RouterProvider
        router={router}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      />
    </AuthProvider>
  );
}

export default App;
