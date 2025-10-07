import os
import ssl
import json
import math
import urllib.request
import urllib.error
from flask import Flask, request, render_template_string, url_for
import requests

# ---------- TLS helper for local testing ----------
def allowSelfSignedHttps(allowed: bool):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
allowSelfSignedHttps(True)

# ---------- Azure ML endpoint ----------
AML_ENDPOINT_URL = os.getenv("AML_ENDPOINT_URL", "https://capstonelap-zguaz.canadacentral.inference.ml.azure.com/score")
AML_ENDPOINT_KEY = os.getenv("AML_ENDPOINT_KEY", "3Zj1I9UsG9aPjINySUJwMNy6wyAAo8qAh43fJ67VYvFc2sJSN0BSJQQJ99BJAAAAAAAAAAAAINFRAZML4Ync")

# ---------- FULL installer dataset (50) ----------
INSTALLERS = [
    {"name": "Titan Window Films Ltd. (Victoria, BC)", "lat": 48.435299, "lng": -123.491242},
    {"name": "Titan Window Films Ltd. (Vancouver, BC)", "lat": 49.273114, "lng": -123.100348},
    {"name": "TINTâ€™D Film & Graphic Solutions (Langley, BC)", "lat": 49.112540, "lng": -122.653360},
    {"name": "LR Window Films (North Vancouver, BC)", "lat": 49.304840, "lng": -122.958268},
    {"name": "Urban Window Films (Kelowna, BC)", "lat": 49.880000, "lng": -119.440000},
    {"name": "Signwriter (Cranbrook, BC)", "lat": 49.523389, "lng": -115.761820},
    {"name": "SignTek Industries (Prince George, BC)", "lat": 53.913970, "lng": -122.735200},
    {"name": "Carbon Graphics Group (Edmonton, AB)", "lat": 53.543300, "lng": -113.500700},
    {"name": "Royal Glass (Sylvan Lake, AB)", "lat": 52.311000, "lng": -114.100000},
    {"name": "South Country Glass Ltd. (Medicine Hat, AB)", "lat": 50.039000, "lng": -110.674000},
    {"name": "ATG Architectural Tint & Graphics (Saskatoon, SK)", "lat": 52.174000, "lng": -106.647000},
    {"name": "Artek Film Solutions (Saskatoon, SK)", "lat": 52.130000, "lng": -106.660000},
    {"name": "D&D Sign & Graphic (Regina, SK)", "lat": 50.433000, "lng": -104.500000},
    {"name": "Artek Film Solutions (Regina, SK)", "lat": 50.454000, "lng": -104.618000},
    {"name": "VBG Distributors Ltd. (Winnipeg, MB)", "lat": 49.884000, "lng": -97.058000},
    {"name": "Total Tint (Toronto, ON)", "lat": 43.602000, "lng": -79.545000},
    {"name": "Evolution Window Films (Grimsby, ON)", "lat": 43.195000, "lng": -79.557000},
    {"name": "Peak Window Films (Burlington, ON)", "lat": 43.308000, "lng": -79.855000},
    {"name": "Lindian Enterprises Ltd. (Ottawa, ON)", "lat": 45.334000, "lng": -75.805000},
    {"name": "Capital Solar and Security (Ottawa, ON)", "lat": 45.423000, "lng": -75.610000},
    {"name": "Franklin Tint (Oshawa, ON)", "lat": 43.885000, "lng": -78.856000},
    {"name": "TriCounty Window Film Solutions (Woodstock, ON)", "lat": 43.152000, "lng": -80.754000},
    {"name": "TriCounty Window Film Solutions (Cambridge, ON)", "lat": 43.369000, "lng": -80.312000},
    {"name": "Peak Window Films (Kitchener, ON)", "lat": 43.434000, "lng": -80.472000},
    {"name": "Blissful Blinds Inc. (Goderich, ON)", "lat": 43.742000, "lng": -81.707000},
    {"name": "Unique Window Films (Barrie, ON)", "lat": 44.428000, "lng": -79.664000},
    {"name": "Crystalâ€™s Glass Tinting (Barrie, ON)", "lat": 44.389000, "lng": -79.708000},
    {"name": "Glass Canada Ltd. (London, ON)", "lat": 42.935000, "lng": -81.248000},
    {"name": "Windsor Window Imaging Inc. (Windsor, ON)", "lat": 42.276000, "lng": -83.061000},
    {"name": "Blue Coast Architectural Finishes (Sarnia, ON)", "lat": 42.974000, "lng": -82.406000},
    {"name": "Verticals Nâ€™ Visions (Thunder Bay, ON)", "lat": 48.389000, "lng": -89.244000},
    {"name": "Price Window Films (North Bay, ON)", "lat": 46.311000, "lng": -79.468000},
    {"name": "Sudbury Window Tinting (Sudbury, ON)", "lat": 46.540000, "lng": -80.882000},
    {"name": "Jet Signs (Kingston, ON)", "lat": 44.248000, "lng": -76.571000},
    {"name": "Smart Grafix (Timmins, ON)", "lat": 48.460000, "lng": -81.339000},
    {"name": "Glass Employees Ltd. (Sault Ste. Marie, ON)", "lat": 46.526000, "lng": -84.300000},
    {"name": "Shade Window Films Inc. (Augusta, ON)", "lat": 44.738000, "lng": -75.546000},
    {"name": "Berkayly (MontrÃ©al, QC)", "lat": 45.588000, "lng": -73.612000},
    {"name": "Techteinte BÃ¢timent (Laval, QC)", "lat": 45.600000, "lng": -73.791000},
    {"name": "Stiick Pellicule sur fenÃªtre (Repentigny, QC)", "lat": 45.744000, "lng": -73.443000},
    {"name": "Berkayly (Shefford, QC)", "lat": 45.369000, "lng": -72.538000},
    {"name": "Protech-Sol (Charlesbourg, QC)", "lat": 46.876000, "lng": -71.274000},
    {"name": "Lindian Enterprises Ltd. (Gatineau, QC)", "lat": 45.484000, "lng": -75.641000},
    {"name": "Capital Solar and Security (Gatineau, QC)", "lat": 45.429000, "lng": -75.803000},
    {"name": "Lettrage Express (Chicoutimi-Nord, QC)", "lat": 48.460000, "lng": -71.065000},
    {"name": "Vitrerie KRT (RiviÃ¨re-du-Loup, QC)", "lat": 47.843000, "lng": -69.533000},
    {"name": "Maritime Window Film Specialists (Moncton, NB)", "lat": 46.087000, "lng": -64.811000},
    {"name": "Leonard Film and Graphics (Saint John, NB)", "lat": 45.292000, "lng": -66.037000},
    {"name": "Just Add Color Inc. (Halifax, NS)", "lat": 44.706000, "lng": -63.661000},
    {"name": "Tucker Window Films (St. Johnâ€™s, NL)", "lat": 47.570000, "lng": -52.722000}
]

# ---------- Utilities ----------
def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (math.sin(d_lat/2)**2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lng/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def geocode_address(address, city, province):
    query = f"{address}, {city}, {province}, Canada"
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': query, 'format': 'json', 'addressdetails': 1, 'limit': 1}
    headers = {'User-Agent': 'LAPv2/1.0 (contact: you@example.com)'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 200 and r.json():
            j = r.json()[0]
            return float(j['lat']), float(j['lon'])
    except Exception as e:
        print("Geocoding error:", e)
    return None, None

# ---------- Templates ----------
BASE_HTML_START = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
</head>
<body>
    <!-- Header with Logo and Navigation -->
    <header>
        <div class="header-container">
            <div class="logo-section">
                <img src="{{ url_for('static', filename='windowfilmcanada_logo.jpeg') }}" alt="Window Film Canada Logo" class="company-logo">
            </div>
            <nav class="main-nav">
                <ul class="nav-links">
                    <li><a href="https://www.windowfilmcanada.ca/3m-commercial-solutions">3M Commercial Solutions</a></li>
                    <li><a href="https://www.windowfilmcanada.ca/residential-solutions">3M Residential Solutions</a></li>
                    <li><a href="https://www.windowfilmcanada.ca/graphic-and-architectural-solutions">3M Graphic & Architectural Solutions</a></li>
                    <li><a href="https://www.windowfilmcanada.ca/project-gallery">Project Gallery</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container">
"""

BASE_HTML_END = """
    </main>

    <!-- Footer -->
    <footer>
        <div class="footer-content">
            <div class="footer-section">
                <h3>About</h3>
                <ul>
                    <li><a href="https://www.windowfilmcanada.ca/about-window-film-canada">About Us</a></li>
                    <li><a href="https://www.windowfilmcanada.ca/about-window-film-canada">Services</a></li>
                </ul>
            </div>

            <div class="footer-section">
                <h3>Contact Info</h3>
                <ul>
                    <li><a href="mailto:info@windowfilmcanada.com">info@windowfilmcanada.com</a></li>
                    <li>+1-888-267-3206</li>
                    <li>325 Watline, Aevnue, Mississauga, ON</li>
                </ul>
            </div>

            <div class="footer-section">
                <h3>Follow Us</h3>
                <div class="social-links">
                    <a href="#facebook" aria-label="Facebook"><i class="fab fa-facebook-f"></i></a>
                    <a href="#twitter" aria-label="Twitter"><i class="fab fa-twitter"></i></a>
                    <a href="#instagram" aria-label="Instagram"><i class="fab fa-instagram"></i></a>
                    <a href="#linkedin" aria-label="LinkedIn"><i class="fab fa-linkedin-in"></i></a>
                </div>
            </div>

            <div class="footer-section">
                <h3>Legal</h3>
                <ul>
                    <li><a href="#privacy">Privacy Policy</a></li>
                </ul>
            </div>
        </div>

        <div class="footer-bottom">
            <p>&copy; 2025 Window Film Canada. All rights reserved.</p>
        </div>
    </footer>

    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
"""

FORM_HTML = BASE_HTML_START + """
        <section class="contact-section">
            <h1>Request a Quote</h1>
            <p class="subtitle">Get in touch with Window Film Canada. We're here to help with all your window film needs.</p>
            <p class="subtitle">We offer consulting, design and installation across Canada.</p>

            <form id="contactForm" class="contact-form" method="POST" action="/predict">
                <div class="form-group">
                    <label for="firstName">First Name *</label>
                    <input type="text" id="firstName" name="first_name" required>
                </div>

                <div class="form-group">
                    <label for="lastName">Last Name *</label>
                    <input type="text" id="lastName" name="last_name" required>
                </div>

                <div class="form-group">
                    <label for="companyName">Company Name</label>
                    <input type="text" id="companyName" name="company_name" placeholder="e.g., Evolution Window Films">
                </div>

                <div class="form-group">
                    <label for="email">Email *</label>
                    <input type="email" id="email" name="email" required>
                </div>

                <div class="form-group">
                    <label for="phone">Phone</label>
                    <input type="tel" id="phone" name="phone">
                </div>

                <div class="form-group">
                    <label for="address">Address *</label>
                    <input type="text" id="address" name="address1" required>
                </div>

                <div class="form-group">
                    <label for="city">City *</label>
                    <input type="text" id="city" name="city" required>
                </div>

                <div class="form-group">
                    <label for="province">Province/State *</label>
                    <select id="province" name="province" required>
                        <option value="">Select Province</option>
                        <option value="AB">Alberta</option>
                        <option value="BC">British Columbia</option>
                        <option value="MB">Manitoba</option>
                        <option value="NB">New Brunswick</option>
                        <option value="NL">Newfoundland and Labrador</option>
                        <option value="NT">Northwest Territories</option>
                        <option value="NS">Nova Scotia</option>
                        <option value="NU">Nunavut</option>
                        <option value="ON">Ontario</option>
                        <option value="PE">Prince Edward Island</option>
                        <option value="QC">Quebec</option>
                        <option value="SK">Saskatchewan</option>
                        <option value="YT">Yukon</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="postal">Postal Code</label>
                    <input type="text" id="postal" name="postal" placeholder="e.g., M5V 2T6">
                </div>

                <div class="form-group">
                    <label for="interest">I'm Interested In *</label>
                    <select id="interest" name="interest" required>
                        <option value="">Select an option</option>
                        <option value="commercial">Commercial</option>
                        <option value="residential">Residential</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="comments">Comments</label>
                    <textarea id="comments" name="comments" rows="5"></textarea>
                </div>

                <button type="submit" class="submit-btn">Request a quote</button>
            </form>

            <div id="successMessage" class="success-message" style="display: none;">
                Thank you for contacting us! We'll get back to you shortly.
            </div>
        </section>
""" + BASE_HTML_END

SELECT_INSTALLER_HTML = BASE_HTML_START + """
        <section class="contact-section">
            <h1>Select the Nearest Installer</h1>
            {% if installers %}
                <p class="subtitle">Installers within {{ radius }} km:</p>
                <form method="POST" action="/finalize" class="contact-form">
                    <div class="form-group">
                        <label for="installer">Choose Installer</label>
                        <select id="installer" name="installer">
                            {% for inst in installers %}
                                <option value="{{ inst['name'] }}">{{ inst['name'] }} ({{ inst['distance']|round(2) }} km)</option>
                            {% endfor %}
                        </select>
                    </div>

                    <!-- pass-through fields -->
                    <input type="hidden" name="first_name" value="{{ contact_data.first_name }}">
                    <input type="hidden" name="last_name" value="{{ contact_data.last_name }}">
                    <input type="hidden" name="email" value="{{ contact_data.email }}">
                    <input type="hidden" name="company_name" value="{{ contact_data.company_name }}">
                    <input type="hidden" name="address1" value="{{ contact_data.address1 }}">
                    <input type="hidden" name="city" value="{{ contact_data.city }}">
                    <input type="hidden" name="province" value="{{ contact_data.province }}">
                    <input type="hidden" name="postal" value="{{ contact_data.postal }}">
                    <input type="hidden" name="user_lat" value="{{ user_lat }}">
                    <input type="hidden" name="user_lng" value="{{ user_lng }}">
                    <input type="hidden" name="radius" value="{{ radius }}">

                    <button class="submit-btn" type="submit">Confirm Installer</button>
                </form>
            {% else %}
                <div class="alert-warn" style="background-color:#fff3cd;color:#856404;padding:1rem;border-radius:5px;border:1px solid #ffeaa7;margin:1.5rem 0;text-align:center;">
                    No installers found within {{ radius }} km.
                </div>
                <form method="POST" action="/expand_search" class="contact-form">
                    <input type="hidden" name="first_name" value="{{ contact_data.first_name }}">
                    <input type="hidden" name="last_name" value="{{ contact_data.last_name }}">
                    <input type="hidden" name="email" value="{{ contact_data.email }}">
                    <input type="hidden" name="company_name" value="{{ contact_data.company_name }}">
                    <input type="hidden" name="address1" value="{{ contact_data.address1 }}">
                    <input type="hidden" name="city" value="{{ contact_data.city }}">
                    <input type="hidden" name="province" value="{{ contact_data.province }}">
                    <input type="hidden" name="postal" value="{{ contact_data.postal }}">
                    <input type="hidden" name="user_lat" value="{{ user_lat }}">
                    <input type="hidden" name="user_lng" value="{{ user_lng }}">
                    <input type="hidden" name="radius" value="{{ radius }}">
                    <button class="submit-btn" type="submit">Expand Search Radius</button>
                </form>
            {% endif %}
        </section>
""" + BASE_HTML_END

FINAL_HTML = BASE_HTML_START + """
        <section class="contact-section">
            <h1>Installer Selected</h1>
            <div class="success-message">
                Thank you <strong>{{ first_name }} {{ last_name }}</strong>. You selected <strong>{{ installer_name }}</strong>.
            </div>

            <h2 style="color:#dc143c;margin-top:2rem;margin-bottom:1rem;font-size:1.5rem;">Machine Learning Model Output</h2>
            <textarea readonly rows="12" style="width:100%;border:2px solid #ddd;border-radius:5px;padding:0.8rem;font-family:inherit;font-size:1rem;">{{ ml_result }}</textarea>

            <div style="margin-top:1.5rem;">
                <a class="submit-btn" href="/" style="text-decoration:none;display:inline-block;text-align:center;">Go back</a>
            </div>
        </section>
""" + BASE_HTML_END

# ---------- Flask app ----------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET'])
def index():
    return render_template_string(FORM_HTML, title="Request a Quote - Window Film Canada")

@app.route('/predict', methods=['POST'])
def predict():
    contact = {
        "first_name": request.form.get('first_name'),
        "last_name": request.form.get('last_name'),
        "email": request.form.get('email'),
        "company_name": request.form.get('company_name'),
        "address1": request.form.get('address1'),
        "city": request.form.get('city'),
        "province": request.form.get('province'),
        "postal": request.form.get('postal'),
    }
    lat, lng = geocode_address(contact["address1"], contact["city"], contact["province"])
    if lat is None or lng is None:
        return "<h3>Could not geocode your address. Please check it and try again.</h3><a href='/'>Back</a>"

    radius = 30.0
    nearby = []
    for inst in INSTALLERS:
        d = haversine_distance(lat, lng, inst["lat"], inst["lng"])
        if d <= radius:
            nearby.append({"name": inst["name"], "distance": d})

    return render_template_string(
        SELECT_INSTALLER_HTML,
        title="Select Installer - Window Film Canada",
        installers=nearby,
        user_lat=lat,
        user_lng=lng,
        radius=radius,
        contact_data=contact
    )

@app.route('/expand_search', methods=['POST'])
def expand_search():
    contact = {
        "first_name": request.form.get('first_name'),
        "last_name": request.form.get('last_name'),
        "email": request.form.get('email'),
        "company_name": request.form.get('company_name'),
        "address1": request.form.get('address1'),
        "city": request.form.get('city'),
        "province": request.form.get('province'),
        "postal": request.form.get('postal'),
    }
    lat = float(request.form.get('user_lat'))
    lng = float(request.form.get('user_lng'))
    prev_radius = float(request.form.get('radius'))
    radius = prev_radius + 20.0

    nearby = []
    for inst in INSTALLERS:
        d = haversine_distance(lat, lng, inst["lat"], inst["lng"])
        if d <= radius:
            nearby.append({"name": inst["name"], "distance": d})

    return render_template_string(
        SELECT_INSTALLER_HTML,
        title="Select Installer - Window Film Canada",
        installers=nearby,
        user_lat=lat,
        user_lng=lng,
        radius=radius,
        contact_data=contact
    )

def _post_aml(url, headers, payload_dict):
    body = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read().decode("utf-8")

@app.route('/finalize', methods=['POST'])
def finalize():
    selected = request.form.get('installer')
    contact = {
        "first_name": request.form.get('first_name'),
        "last_name": request.form.get('last_name'),
        "email": request.form.get('email'),
        "company_name": request.form.get('company_name'),
        "address1": request.form.get('address1'),
        "city": request.form.get('city'),
        "province": request.form.get('province'),
        "postal": request.form.get('postal'),
    }

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if AML_ENDPOINT_KEY:
        headers["Authorization"] = f"Bearer {AML_ENDPOINT_KEY}"

    attempts = [
        ("v2_columns_data", {"input_data": {"columns": list(contact.keys()),
                                            "data": [[contact[k] for k in contact.keys()]]}}),
        ("records", {"input_data": [contact]}),
        ("legacy", {"Inputs": {"data": [contact]}})
    ]

    ml_result, last_err = "", None
    for label, payload in attempts:
        try:
            status, text = _post_aml(AML_ENDPOINT_URL, headers, payload)
            if 200 <= status < 300:
                try:
                    ml_result = f"(format tried: {label})\n" + json.dumps(json.loads(text), indent=2)
                except Exception:
                    ml_result = f"(format tried: {label})\n{text}"
                last_err = None
                break
            else:
                last_err = f"HTTP {status}: {text}"
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode('utf-8','ignore')}"
        except Exception as e:
            last_err = f"Error: {e}"
    if last_err and not ml_result:
        ml_result = last_err

    try:
        with open("installer_selection_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{contact['first_name']} {contact['last_name']} -> {selected}\n")
    except Exception as e:
        print("Log write error:", e)

    return render_template_string(
        FINAL_HTML,
        title="Quote Submitted - Window Film Canada",
        installer_name=selected,
        first_name=contact['first_name'],
        last_name=contact['last_name'],
        ml_result=ml_result
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)