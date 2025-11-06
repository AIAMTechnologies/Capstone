// Form validation and submission handling
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('contactForm');
    const successMessage = document.getElementById('successMessage');

    // Only run if form exists on the page
    if (form) {
        // Phone number formatting
        const phoneInput = document.getElementById('phone');
        if (phoneInput) {
            phoneInput.addEventListener('input', function(e) {
                let value = e.target.value.replace(/\D/g, '');
                
                if (value.length > 0) {
                    if (value.length <= 3) {
                        value = value;
                    } else if (value.length <= 6) {
                        value = `(${value.slice(0, 3)}) ${value.slice(3)}`;
                    } else {
                        value = `(${value.slice(0, 3)}) ${value.slice(3, 6)}-${value.slice(6, 10)}`;
                    }
                }
                
                e.target.value = value;
            });
        }

        // Form submission
        form.addEventListener('submit', function(e) {
            // Get form data
            const formData = {
                firstName: document.getElementById('firstName')?.value || '',
                lastName: document.getElementById('lastName')?.value || '',
                email: document.getElementById('email')?.value || '',
                phone: document.getElementById('phone')?.value || '',
                address: document.getElementById('address')?.value || '',
                city: document.getElementById('city')?.value || '',
                province: document.getElementById('province')?.value || '',
                interest: document.getElementById('interest')?.value || '',
                comments: document.getElementById('comments')?.value || ''
            };

            // Validate required fields
            if (!validateForm(formData)) {
                e.preventDefault(); // Only prevent if validation fails
                return false;
            }

            // If validation passes, let the form submit normally to the server
            // The server will handle the geocoding and installer matching
            // DO NOT call e.preventDefault() here!
        });
    }

    // Form validation function
    function validateForm(data) {
        // Check required fields
        if (!data.firstName.trim()) {
            alert('Please enter your first name');
            return false;
        }

        if (!data.lastName.trim()) {
            alert('Please enter your last name');
            return false;
        }

        if (!data.email.trim()) {
            alert('Please enter your email');
            return false;
        }

        // Validate email format
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(data.email)) {
            alert('Please enter a valid email address');
            return false;
        }

        if (!data.address.trim()) {
            alert('Please enter your address');
            return false;
        }

        if (!data.city.trim()) {
            alert('Please enter your city');
            return false;
        }

        if (!data.province) {
            alert('Please select a province');
            return false;
        }

        if (!data.interest) {
            alert('Please select what you are interested in');
            return false;
        }

        return true;
    }

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            // Only prevent default for hash links that aren't meant to be external
            if (href !== '#') {
                e.preventDefault();
                
                // If the target exists on the page, scroll to it
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // Add animation to form inputs on focus
    const inputs = document.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.style.transform = 'scale(1.02)';
            this.style.transition = 'transform 0.2s ease';
        });

        input.addEventListener('blur', function() {
            this.style.transform = 'scale(1)';
        });
    });
});