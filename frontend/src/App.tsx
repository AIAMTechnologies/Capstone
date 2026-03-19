import React from 'react';
import { createBrowserRouter, RouterProvider, Navigate, Outlet } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import Header from './components/Header';
import ProtectedRoute from './components/ProtectedRoute';
import Home from './pages/Home';
import AdminLogin from './pages/AdminLogin';
import AdminLayout from './components/admin/AdminLayout';
import Dashboard from './pages/admin/Dashboard';
import History from './pages/admin/History';
import Resources from './pages/admin/Resources';
import Reports from './pages/admin/Reports';
import Tools from './pages/admin/Tools';
import './styles/App.css';

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
        { path: 'dashboard', element: <Dashboard /> },
        { path: 'history', element: <History /> },
        { path: 'resources', element: <Resources /> },
        { path: 'reports', element: <Reports /> },
        { path: 'tools', element: <Tools /> },
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
