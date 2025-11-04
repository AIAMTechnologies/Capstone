import React from 'react';
import ContactForm from '../components/ContactForm';
import { Shield, Award, Clock, MapPin } from 'lucide-react';

const Home: React.FC = () => {
  return (
    <div>
      {/* Hero Section */}
      <section style={styles.hero}>
        <div className="container">
          <div style={styles.heroContent}>
            <h1 style={styles.heroTitle}>
              Professional Window Film Installation
            </h1>
            <p style={styles.heroSubtitle}>
              Protect your home or business with premium window film solutions across Canada
            </p>
            <div style={styles.features}>
              <div style={styles.feature}>
                <Shield size={24} />
                <span>UV Protection</span>
              </div>
              <div style={styles.feature}>
                <Award size={24} />
                <span>Certified Installers</span>
              </div>
              <div style={styles.feature}>
                <Clock size={24} />
                <span>Quick Installation</span>
              </div>
              <div style={styles.feature}>
                <MapPin size={24} />
                <span>Nationwide Service</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Form Section */}
      <section style={styles.formSection}>
        <div className="container">
          <ContactForm />
        </div>
      </section>

      {/* Benefits Section */}
      <section style={styles.benefits}>
        <div className="container">
          <h2 style={styles.sectionTitle}>Why Choose Window Film Canada?</h2>
          <div className="grid grid-3">
            <div className="card">
              <div style={styles.benefitIcon}>
                <Shield size={48} color="#c91414" />
              </div>
              <h3>Superior Protection</h3>
              <p>
                Block up to 99% of harmful UV rays while reducing heat and glare. 
                Protect your furnishings and reduce energy costs.
              </p>
            </div>
            <div className="card">
              <div style={styles.benefitIcon}>
                <Award size={48} color="#c91414" />
              </div>
              <h3>Expert Installation</h3>
              <p>
                Our certified installers have years of experience and use only 
                premium materials for lasting results.
              </p>
            </div>
            <div className="card">
              <div style={styles.benefitIcon}>
                <Clock size={48} color="#c91414" />
              </div>
              <h3>Fast Service</h3>
              <p>
                Get matched with a local installer quickly. Most installations 
                are completed within 1-2 days.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={styles.footer}>
        <div className="container">
          <div className="grid grid-3">
            <div>
              <h4 style={styles.footerHeading}>Window Film Canada</h4>
              <p style={styles.footerText}>
                Professional window film installation services across Canada
              </p>
            </div>
            <div>
              <h4 style={styles.footerHeading}>Services</h4>
              <ul style={styles.footerList}>
                <li>Residential Window Film</li>
                <li>Commercial Window Film</li>
                <li>UV Protection</li>
                <li>Energy Efficiency</li>
              </ul>
            </div>
            <div>
              <h4 style={styles.footerHeading}>Contact</h4>
              <p style={styles.footerText}>
                Email: info@windowfilmcanada.com<br />
                Phone: 1-800-XXX-XXXX
              </p>
            </div>
          </div>
          <div style={styles.copyright}>
            <p>© 2025 Window Film Canada. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

const styles = {
  hero: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    padding: '80px 0',
    textAlign: 'center' as const,
  },
  heroContent: {
    maxWidth: '900px',
    margin: '0 auto',
  },
  heroTitle: {
    fontSize: '48px',
    fontWeight: '700' as const,
    marginBottom: '20px',
    lineHeight: '1.2',
  },
  heroSubtitle: {
    fontSize: '20px',
    marginBottom: '40px',
    opacity: 0.9,
  },
  features: {
    display: 'flex',
    justifyContent: 'center',
    gap: '32px',
    flexWrap: 'wrap' as const,
  },
  feature: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '16px',
    fontWeight: '600' as const,
  },
  formSection: {
    padding: '40px 0',
    backgroundColor: '#f5f5f5',
  },
  benefits: {
    padding: '80px 0',
    backgroundColor: 'white',
  },
  sectionTitle: {
    fontSize: '36px',
    fontWeight: '700' as const,
    textAlign: 'center' as const,
    marginBottom: '48px',
    color: '#2c3e50',
  },
  benefitIcon: {
    marginBottom: '16px',
  },
  footer: {
    backgroundColor: '#2c3e50',
    color: 'white',
    padding: '48px 0 24px',
  },
  footerHeading: {
    fontSize: '18px',
    fontWeight: '700' as const,
    marginBottom: '16px',
  },
  footerText: {
    fontSize: '14px',
    lineHeight: '1.8',
    opacity: 0.8,
  },
  footerList: {
    listStyle: 'none',
    padding: 0,
    margin: 0,
  },
  copyright: {
    textAlign: 'center' as const,
    marginTop: '32px',
    paddingTop: '24px',
    borderTop: '1px solid rgba(255,255,255,0.1)',
    opacity: 0.6,
  },
};

export default Home;