# 🏗️ Rectangular RCC Hopper Bunker Design Software

An interactive engineering and structural design software for **Rectangular Reinforced Concrete (RCC) Hopper Bunkers**, developed in compliance with **Indian Standards (IS 4995 & IS 456)** and deployed using **Streamlit**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Features

1. **Geometry & Capacity Engine:**
   - Accurate volume calculations for rectangular pyramidal hopper frustums and vertical containment walls.
   - Self-weight estimation of walls, hopper slabs, and stored bulk material.
   - Interactive 3D geometry visualization using Plotly 3D.

2. **Vertical Wall Structural Analysis:**
   - Multiple support conditions (Cantilever, Simply Supported, Fixed at both ends).
   - Real-time generation of Shear Force Diagrams (SFD) and Bending Moment Diagrams (BMD).

3. **Hopper Slab Structural Analysis:**
   - Inclined panel pressure distribution under varying normal and tangential loading.
   - Slope angle resolution and panel-by-panel reaction calculation.

4. **Outlet Ring Beams & Corner Columns:**
   - Design of 4 bottom ring beams supporting hopper gravity reaction and self-weight.
   - Axial load transfer to 4 corner columns under limit state factored combinations.

5. **IS 456:2000 RCC Design & Code Verification:**
   - Singly reinforced flexural design for slabs and beams.
   - Shear stress checks ($\tau_v$, $\tau_c$ from Table 19, $\tau_{c,max}$ from Table 20).
   - Automated compliance checklist (Pass/Fail) for spacing, minimum steel, and neutral axis depths.

6. **Bar Bending Schedule (BBS) & Rebar Takeoff:**
   - Automated cutting length and quantity computation per member.
   - Steel takeoff summary grouped by diameter ($d^2/162$).
   - One-click export to CSV.

7. **2D & 3D Structural Detailing:**
   - 2D Plan View with corner hip bars and outlet dimensions.
   - 2D Elevation/Section View detailing column, wall, and hopper rebar.
   - Full 3D interactive wireframe rebar cage model.

8. **Automated Calculation Report & PDF Export:**
   - Complete design calculations exportable to professional PDF via ReportLab.

---

## 🚀 Deployment to Streamlit Cloud

To deploy this repository to **Streamlit Community Cloud** (free hosting):

1. **Push this repository to GitHub:**
   Make sure this repository is pushed to your GitHub account:
   ```bash
   git add .
   git commit -m "Deploy RCC Hopper Bunker Design Software to Streamlit"
   git push origin main
   ```

2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your repository: `lakshmigowthami6/civil-measurements`.
   - Branch: `main` (or `master`).
   - Main file path: `app.py`.
   - Click **"Deploy!"**.

Your application will be live in minutes with a shareable public URL!

---

## 💻 Local Installation & Setup

If you want to run the application locally:

### 1. Clone the repository
```bash
git clone https://github.com/lakshmigowthami6/civil-measurements.git
cd civil-measurements
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
civil-measurements/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies for Streamlit Cloud
├── README.md                  # Project documentation & deployment guide
├── .gitignore                 # Ignored files and caches
└── .streamlit/
    └── config.toml            # Streamlit theme and UI configuration
```

---

## 📚 References & Design Framework

- **IS 4995 (Part 1): 1974** – Criteria for design of reinforced concrete bins for the storage of granular and powdery materials: Part 1 General requirements and assessment of bin loads.
- **IS 4995 (Part 2): 1974** – Design criteria for reinforced concrete bins.
- **IS 456: 2000** – Code of practice for plain and reinforced concrete (Fourth Revision).
- **SP 16** – Design Aids for Reinforced Concrete to IS 456:1978.

---

## ⚠️ Engineering Disclaimer

*This application is an educational and preliminary engineering sizing tool. Final structural designs, rebar schedules, and construction drawings for critical storage bins must be independently verified by a licensed structural engineer with certified structural analysis software in accordance with project-specific soil, seismic, wind, and material conditions.*
