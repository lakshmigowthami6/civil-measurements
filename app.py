"""
Rectangular RCC Hopper Bunker – Design Software (IS 4995 & IS 456)
Streamlit Web Application
"""

import math
import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Try importing reportlab for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RCC Hopper Bunker Design Software",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #E0E7FF;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        margin: 8px 0 0 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ENGINEERING ENGINE FUNCTIONS
# -----------------------------------------------------------------------------
def frustum_volume_rectangular(Lt, Wt, Lb, Wb, H):
    A1, A2 = Lt * Wt, Lb * Wb
    return H / 3.0 * (A1 + A2 + math.sqrt(max(A1 * A2, 0.0)))

def sloping_panel_area(top_a, bottom_a, height):
    return 0.5 * (top_a + bottom_a) * math.sqrt(height**2 + ((top_a - bottom_a) / 2.0)**2)

def bunker_geometry(inp):
    Li = inp["L_outer"] - 2 * inp["t_wall"]
    Wi = inp["W_outer"] - 2 * inp["t_wall"]
    if Li <= 0 or Wi <= 0:
        raise ValueError("Wall thickness leaves no positive internal plan dimensions.")
    if inp["L_outlet"] <= 0 or inp["W_outlet"] <= 0:
        raise ValueError("Outlet dimensions must be positive.")
    if inp["L_outlet"] >= Li or inp["W_outlet"] >= Wi:
        raise ValueError("Outlet dimensions must be smaller than internal hopper-top dimensions.")
    if inp["H_wall"] <= 0 or inp["H_hopper"] <= 0:
        raise ValueError("Wall and hopper heights must be positive.")

    V_vertical = Li * Wi * inp["H_wall"]
    V_hopper = frustum_volume_rectangular(Li, Wi, inp["L_outlet"], inp["W_outlet"], inp["H_hopper"])
    V_total = V_vertical + V_hopper

    A_hopper = (
        2 * sloping_panel_area(Wi, inp["W_outlet"], inp["H_hopper"]) +
        2 * sloping_panel_area(Li, inp["L_outlet"], inp["H_hopper"])
    )
    A_wall = 2 * (Li + Wi) * inp["H_wall"]
    A_base = inp["L_outlet"] * inp["W_outlet"]

    return {
        "L_internal": Li, "W_internal": Wi,
        "A_internal": Li * Wi,
        "V_vertical": V_vertical, "V_hopper": V_hopper,
        "V_capacity": V_total, "A_vertical_walls": A_wall,
        "A_hopper_sloping": A_hopper, "A_outlet": A_base
    }

def calculate_stage1(inp):
    g = bunker_geometry(inp)
    gamma_c = inp["gamma_concrete"]
    gamma_m = inp["gamma_material"]

    V_walls_conc = g["A_vertical_walls"] * inp["t_wall"]
    V_hopper_conc = g["A_hopper_sloping"] * inp["t_hopper"]
    W_walls = V_walls_conc * gamma_c
    W_hopper = V_hopper_conc * gamma_c
    W_material = g["V_capacity"] * gamma_m
    q_equiv = W_material / max(g["A_outlet"], 1e-9)

    p_screen_base = inp["K_screen"] * gamma_m * inp["H_wall"]
    M_screen_per_m = p_screen_base * inp["H_wall"]**2 / 6.0

    return {
        **g,
        "V_walls_concrete": V_walls_conc,
        "V_hopper_concrete": V_hopper_conc,
        "W_walls": W_walls,
        "W_hopper": W_hopper,
        "W_material": W_material,
        "W_concrete_total": W_walls + W_hopper,
        "W_total_simple": W_material + W_walls + W_hopper,
        "q_equiv_outlet": q_equiv,
        "p_screen_base": p_screen_base,
        "M_screen_per_m": M_screen_per_m
    }

def perform_wall_analysis(H, q, support):
    b = 1.0
    x = np.linspace(0, H, 150)
    if support == "cantilever":
        V = q * (H - x) * b
        M = q * (H - x)**2 * b / 2
        R_bottom = q * H * b
        R_top = 0.0
        M_max = q * H**2 * b / 2
        M_min = 0.0
    elif support == "simply":
        V = q * (H / 2 - x) * b
        M = q * x * (H - x) * b / 2
        R_bottom = q * H * b / 2
        R_top = q * H * b / 2
        M_max = q * H**2 * b / 8
        M_min = 0.0
    else:  # fixed
        V = q * (H / 2 - x) * b
        M = (-q * H**2 / 12 + q * H * x / 2 - q * x**2 / 2) * b
        R_bottom = q * H * b / 2
        R_top = q * H * b / 2
        M_max = q * H**2 * b / 24
        M_min = -q * H**2 * b / 12

    return {
        "x": x, "V": V, "M": M,
        "R_bottom": R_bottom, "R_top": R_top,
        "M_max": M_max, "M_min": M_min
    }

def calculate_hopper_face_actions(inp):
    Li = inp["L_outer"] - 2 * inp["t_wall"]
    Wi = inp["W_outer"] - 2 * inp["t_wall"]
    Lo = inp["L_outlet"]
    Wo = inp["W_outlet"]
    H = inp["H_wall"]
    Hh = inp["H_hopper"]
    gamma = inp["gamma_material"]
    gamma_c = inp["gamma_concrete"]
    K = inp["K_screen"]
    th = inp["t_hopper"]

    faces = [
        {"Face": "Longitudinal face 1", "Run": (Wi - Wo) / 2, "Outlet edge": Lo},
        {"Face": "Longitudinal face 2", "Run": (Wi - Wo) / 2, "Outlet edge": Lo},
        {"Face": "Transverse face 1", "Run": (Li - Lo) / 2, "Outlet edge": Wo},
        {"Face": "Transverse face 2", "Run": (Li - Lo) / 2, "Outlet edge": Wo}
    ]

    actions = []
    for face in faces:
        run = face["Run"]
        edge = face["Outlet edge"]
        S = np.sqrt(Hh**2 + run**2)
        theta = np.arctan2(Hh, run)

        pv_top = gamma * H
        pv_bottom = gamma * (H + Hh)
        ph_top = K * pv_top
        ph_bottom = K * pv_bottom

        q_top = pv_top * np.cos(theta) + ph_top * np.sin(theta)
        q_bottom = pv_bottom * np.cos(theta) + ph_bottom * np.sin(theta)

        # Hopper slab self-weight normal component
        q_self = gamma_c * th * np.cos(theta)
        q_top += q_self
        q_bottom += q_self

        s = np.linspace(0, S, 501)
        q = q_top + (q_bottom - q_top) * s / S
        ds = np.diff(s)

        load_segments = ((q[:-1] + q[1:]) / 2) * ds
        total_load = np.sum(load_segments)
        moment_segments = ((q[:-1] * s[:-1] + q[1:] * s[1:]) / 2) * ds
        total_moment = np.sum(moment_segments)

        R_outlet = total_moment / S
        R_top = total_load - R_outlet

        cumulative_load = np.zeros(len(s))
        cumulative_load[1:] = np.cumsum(load_segments)
        V = R_top - cumulative_load

        M = np.zeros(len(s))
        M[1:] = np.cumsum(((V[:-1] + V[1:]) / 2) * ds)

        actions.append({
            "Face": face["Face"],
            "Run": run,
            "Slope length": S,
            "Outlet edge": edge,
            "Angle": np.degrees(theta),
            "Top pressure": q_top,
            "Bottom pressure": q_bottom,
            "Outlet reaction": R_outlet,
            "Top reaction": R_top,
            "Max moment": np.max(np.abs(M)),
            "Max shear": np.max(np.abs(V)),
            "s": s,
            "V": V,
            "M": M
        })
    return actions

def bar_area(dia):
    return np.pi * dia**2 / 4.0

def required_flexural_steel(Mu_kNm, b, d, fck, fy):
    Mu = Mu_kNm * 1e6
    if Mu <= 0:
        return 0.0
    alpha = 0.42 * 0.87 * fy / (0.36 * fck * b)
    disc = d**2 - 4 * alpha * Mu / (0.87 * fy)
    if disc < 0:
        raise ValueError(f"Moment {Mu_kNm:.2f} kNm exceeds singly reinforced section capacity.")
    return (d - np.sqrt(disc)) / (2 * alpha)

def limiting_steel_ratio(fy):
    if fy <= 250:
        return 0.53
    elif fy <= 415:
        return 0.48
    else:
        return 0.46

# IS 456 Table 19 & 20
PT_VALUES = np.array([0.15, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00])
FCK_GRADES = np.array([15, 20, 25, 30, 35, 40])
TAU_C_TABLE = np.array([
    [0.28, 0.28, 0.29, 0.29, 0.29, 0.30],
    [0.35, 0.36, 0.36, 0.37, 0.37, 0.38],
    [0.46, 0.48, 0.49, 0.50, 0.50, 0.51],
    [0.54, 0.56, 0.57, 0.59, 0.59, 0.60],
    [0.60, 0.62, 0.64, 0.66, 0.67, 0.68],
    [0.64, 0.67, 0.70, 0.71, 0.73, 0.74],
    [0.68, 0.72, 0.74, 0.76, 0.78, 0.79],
    [0.71, 0.75, 0.78, 0.80, 0.82, 0.84],
    [0.71, 0.79, 0.82, 0.84, 0.86, 0.88],
    [0.71, 0.81, 0.85, 0.88, 0.90, 0.92],
    [0.71, 0.82, 0.88, 0.91, 0.93, 0.95],
    [0.71, 0.82, 0.90, 0.94, 0.96, 0.98],
    [0.71, 0.82, 0.92, 0.96, 0.99, 1.01]
])
TAU_C_MAX_TABLE = np.array([2.50, 2.80, 3.10, 3.50, 3.70, 4.00])

def get_tau_c(pt, fck_val):
    fck_val = max(15, min(fck_val, 40))
    pt = max(0.15, min(float(pt), 3.0))
    values_at_grades = np.array([np.interp(pt, PT_VALUES, TAU_C_TABLE[:, j]) for j in range(len(FCK_GRADES))])
    return float(np.interp(fck_val, FCK_GRADES, values_at_grades))

def get_tau_c_max(fck_val):
    fck_val = max(15, min(fck_val, 40))
    return float(np.interp(fck_val, FCK_GRADES, TAU_C_MAX_TABLE))

def steel_unit_weight(diameter):
    return diameter**2 / 162.0

def number_of_bars(length_mm, spacing_mm):
    return max(2, int(math.ceil(length_mm / max(spacing_mm, 1e-3))) + 1)

def make_bbs_row(mark, member, bar_type, diameter, quantity, cutting_length):
    total_length = quantity * cutting_length / 1000.0
    unit_w = steel_unit_weight(diameter)
    total_w = total_length * unit_w
    return {
        "Bar mark": mark,
        "Member": member,
        "Bar type": bar_type,
        "Diameter (mm)": diameter,
        "Quantity": quantity,
        "Cutting length (mm)": round(cutting_length, 1),
        "Total length (m)": round(total_length, 3),
        "Unit weight (kg/m)": round(unit_w, 3),
        "Total weight (kg)": round(total_w, 3)
    }

# -----------------------------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/silo.png", width=64)
    st.title("Bunker Parameters")

    with st.expander("📐 1. Bunker Geometry", expanded=True):
        L_outer = st.number_input("Outer length L (m)", min_value=2.0, max_value=30.0, value=6.0, step=0.5)
        W_outer = st.number_input("Outer width W (m)", min_value=2.0, max_value=30.0, value=4.0, step=0.5)
        H_wall = st.number_input("Vertical wall height H (m)", min_value=1.0, max_value=25.0, value=5.0, step=0.5)
        t_wall = st.number_input("Wall thickness t (m)", min_value=0.15, max_value=1.0, value=0.25, step=0.05)
        H_hopper = st.number_input("Hopper height Hh (m)", min_value=0.5, max_value=15.0, value=2.0, step=0.25)
        L_outlet = st.number_input("Outlet length Lo (m)", min_value=0.4, max_value=5.0, value=1.0, step=0.2)
        W_outlet = st.number_input("Outlet width Wo (m)", min_value=0.4, max_value=5.0, value=1.0, step=0.2)
        t_hopper = st.number_input("Hopper slab thickness (m)", min_value=0.15, max_value=0.8, value=0.25, step=0.05)

    with st.expander("🧱 2. Stored Material & Concrete", expanded=False):
        material_name = st.text_input("Material Name", value="Coal / Granular aggregate")
        gamma_material = st.number_input("Material Unit Weight γ (kN/m³)", min_value=5.0, max_value=30.0, value=10.0, step=0.5)
        phi_internal = st.number_input("Internal friction angle φ (°)", min_value=10.0, max_value=50.0, value=30.0, step=1.0)
        delta_wall = st.number_input("Wall friction angle δ (°)", min_value=5.0, max_value=40.0, value=20.0, step=1.0)
        gamma_concrete = st.number_input("Concrete Unit Weight γc (kN/m³)", min_value=20.0, max_value=28.0, value=25.0, step=0.5)
        K_screen = st.number_input("Lateral screening factor K", min_value=0.1, max_value=1.0, value=0.33, step=0.05)

    with st.expander("⚙️ 3. Concrete & Steel Grades", expanded=False):
        fck = st.selectbox("Concrete Grade fck (MPa)", options=[20, 25, 30, 35, 40], index=1)
        fy = st.selectbox("Steel Grade fy (MPa)", options=[250, 415, 500, 550], index=1)
        cover_mm = st.number_input("Nominal cover (mm)", min_value=20, max_value=75, value=40, step=5)

    with st.expander("🏛️ 4. Outlet Framing & Columns", expanded=False):
        outlet_beam_b = st.number_input("Outlet beam width b (mm)", min_value=200, max_value=800, value=300, step=50)
        outlet_beam_D = st.number_input("Outlet beam depth D (mm)", min_value=300, max_value=1200, value=450, step=50)
        column_b = st.number_input("Column width (mm)", min_value=200, max_value=1000, value=300, step=50)
        column_D = st.number_input("Column depth (mm)", min_value=200, max_value=1000, value=300, step=50)
        column_h = st.number_input("Column free height (m)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        load_factor = st.number_input("Limit State Load factor (LF)", min_value=1.0, max_value=2.0, value=1.5, step=0.1)

    with st.expander("🔬 5. Rebar Sizing Choices", expanded=False):
        slab_main_dia = st.selectbox("Slab main bar (mm)", [8, 10, 12, 16, 20], index=2)
        slab_dist_dia = st.selectbox("Slab distribution bar (mm)", [8, 10, 12, 16], index=1)
        beam_main_dia = st.selectbox("Beam main bar (mm)", [12, 16, 20, 25, 32], index=1)
        beam_stirrup_dia = st.selectbox("Beam stirrup dia (mm)", [8, 10, 12], index=0)
        beam_stirrup_spacing = st.number_input("Beam stirrup spacing (mm)", value=150, step=25)
        col_bar_dia = st.selectbox("Column main bar (mm)", [16, 20, 25, 32], index=1)
        col_bar_count = st.number_input("Column main bar count", min_value=4, max_value=20, value=8, step=2)
        col_tie_dia = st.selectbox("Column tie dia (mm)", [8, 10, 12], index=0)
        col_tie_spacing = st.number_input("Column tie spacing (mm)", value=150, step=25)
        wall_vert_dia = st.selectbox("Wall vertical bar (mm)", [10, 12, 16, 20], index=2)
        wall_vert_spacing = st.number_input("Wall vertical spacing (mm)", value=150, step=25)
        wall_horiz_dia = st.selectbox("Wall horizontal bar (mm)", [8, 10, 12, 16], index=1)
        wall_horiz_spacing = st.number_input("Wall horizontal spacing (mm)", value=200, step=25)

# Packaging inputs
inp = {
    "L_outer": float(L_outer), "W_outer": float(W_outer),
    "H_wall": float(H_wall), "t_wall": float(t_wall),
    "H_hopper": float(H_hopper), "L_outlet": float(L_outlet),
    "W_outlet": float(W_outlet), "t_hopper": float(t_hopper),
    "gamma_material": float(gamma_material), "phi_internal": float(phi_internal),
    "delta_wall": float(delta_wall), "gamma_concrete": float(gamma_concrete),
    "fck": float(fck), "fy": float(fy), "cover_mm": float(cover_mm),
    "K_screen": float(K_screen), "material_name": material_name
}

# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Rectangular RCC Hopper Bunker – Design Software</h1>
    <p>Comprehensive Structural Analysis, IS 456 Code Verification, Bar Bending Schedule & 2D/3D Detailing</p>
</div>
""", unsafe_allow_html=True)

# Calculation Execution
calc_error = None
try:
    stage1 = calculate_stage1(inp)
    hopper_actions = calculate_hopper_face_actions(inp)
except Exception as e:
    calc_error = str(e)

if calc_error:
    st.error(f"⚠️ Geometry or Parameter Error: {calc_error}")
    st.stop()

# -----------------------------------------------------------------------------
# TOP LEVEL SUMMARY CARDS
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Geometric Capacity</h3>
        <p>{stage1['V_capacity']:.2f} m³</p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Stored Material Weight</h3>
        <p>{stage1['W_material']:.1f} kN</p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Concrete Weight</h3>
        <p>{stage1['W_concrete_total']:.1f} kN</p>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Gravity Action</h3>
        <p>{stage1['W_total_simple']:.1f} kN</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# TAB NAVIGATION
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📐 3D Geometry",
    "📊 Wall Analysis",
    "🌪️ Hopper Slab",
    "🏛️ Beams & Columns",
    "⚙️ IS 456 RCC Design",
    "📋 Bar Schedule (BBS)",
    "🎨 Drawings (2D & 3D)",
    "📄 Report & PDF"
])

# =============================================================================
# TAB 1: GEOMETRY & 3D MODEL
# =============================================================================
with tab1:
    st.subheader("Bunker Geometry & Dynamic 3D Model")
    col_t1_a, col_t1_b = st.columns([1, 1])

    with col_t1_a:
        geom_df = pd.DataFrame([
            ("Internal Length (Li)", f"{stage1['L_internal']:.3f} m"),
            ("Internal Width (Wi)", f"{stage1['W_internal']:.3f} m"),
            ("Vertical Wall Height", f"{inp['H_wall']:.2f} m"),
            ("Wall Thickness", f"{inp['t_wall']:.3f} m"),
            ("Hopper Height", f"{inp['H_hopper']:.2f} m"),
            ("Outlet Opening", f"{inp['L_outlet']:.2f} m × {inp['W_outlet']:.2f} m"),
            ("Vertical Section Storage Volume", f"{stage1['V_vertical']:.2f} m³"),
            ("Hopper Section Storage Volume", f"{stage1['V_hopper']:.2f} m³"),
            ("Total Stored Capacity", f"{stage1['V_capacity']:.2f} m³"),
            ("Screening Lateral Base Pressure", f"{stage1['p_screen_base']:.2f} kN/m²"),
            ("Screening Cantilever Base Moment", f"{stage1['M_screen_per_m']:.2f} kNm/m")
        ], columns=["Property", "Calculated Value"])
        st.dataframe(geom_df, use_container_width=True, hide_index=True)

    with col_t1_b:
        # Plotly 3D Geometry Model
        L = inp["L_outer"]
        W = inp["W_outer"]
        Li = stage1["L_internal"]
        Wi = stage1["W_internal"]
        H = inp["H_wall"]
        Hh = inp["H_hopper"]
        Lo = inp["L_outlet"]
        Wo = inp["W_outlet"]

        fig_geo = go.Figure()

        def add_quad(fig, points, color, opacity=0.85, name=""):
            fig.add_trace(go.Mesh3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=color, opacity=opacity,
                flatshading=True, showscale=False,
                name=name, hoverinfo="name"
            ))

        outer_base = [(-L/2, -W/2, Hh), (L/2, -W/2, Hh), (L/2, W/2, Hh), (-L/2, W/2, Hh)]
        inner_base = [(-Li/2, -Wi/2, Hh), (Li/2, -Wi/2, Hh), (Li/2, Wi/2, Hh), (-Li/2, Wi/2, Hh)]
        outer_top = [(x, y, Hh + H) for x, y, z in outer_base]
        inner_top = [(x, y, Hh + H) for x, y, z in inner_base]
        bottom = [(-Lo/2, -Wo/2, 0), (Lo/2, -Wo/2, 0), (Lo/2, Wo/2, 0), (-Lo/2, Wo/2, 0)]

        for i in range(4):
            j = (i + 1) % 4
            # Outer walls
            add_quad(fig_geo, [outer_base[i], outer_base[j], outer_top[j], outer_top[i]], "#3B82F6", 0.7, "Outer Wall")
            # Inner walls
            add_quad(fig_geo, [inner_base[j], inner_base[i], inner_top[i], inner_top[j]], "#60A5FA", 0.5, "Inner Wall")
            # Top wall rim
            add_quad(fig_geo, [outer_top[i], outer_top[j], inner_top[j], inner_top[i]], "#1D4ED8", 0.9, "Top Rim")
            # Hopper panels
            add_quad(fig_geo, [bottom[i], bottom[j], inner_base[j], inner_base[i]], "#F59E0B", 0.9, "Hopper Slab")

        def draw_edges(fig, points, color="black", width=3):
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                mode="lines",
                line=dict(color=color, width=width),
                showlegend=False
            ))

        draw_edges(fig_geo, outer_top + [outer_top[0]])
        draw_edges(fig_geo, inner_top + [inner_top[0]])
        draw_edges(fig_geo, bottom + [bottom[0]])
        for i in range(4):
            draw_edges(fig_geo, [outer_base[i], outer_top[i]])
            draw_edges(fig_geo, [inner_base[i], inner_top[i]])
            draw_edges(fig_geo, [bottom[i], inner_base[i]])

        fig_geo.update_layout(
            title="Interactive 3D Geometry Model",
            scene=dict(
                xaxis_title="Length (m)",
                yaxis_title="Width (m)",
                zaxis_title="Height (m)",
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=500
        )
        st.plotly_chart(fig_geo, use_container_width=True)

# =============================================================================
# TAB 2: WALL STRUCTURAL ANALYSIS
# =============================================================================
with tab2:
    st.subheader("Vertical Wall Structural Analysis")
    support_cond = st.radio(
        "Select Wall Idealized Boundary Condition:",
        options=["Cantilever", "Simply supported", "Fixed at both ends"],
        horizontal=True
    )
    sup_key = "cantilever" if support_cond == "Cantilever" else ("simply" if support_cond == "Simply supported" else "fixed")

    wall_res = perform_wall_analysis(inp["H_wall"], stage1["p_screen_base"], sup_key)

    rw1, rw2, rw3 = st.columns(3)
    rw1.metric("Bottom Support Reaction", f"{wall_res['R_bottom']:.2f} kN/m")
    rw2.metric("Top Reaction / Beam", f"{wall_res['R_top']:.2f} kN/m")
    rw3.metric("Max Design Moment", f"{wall_res['M_max']:.2f} kNm/m")

    fig_w1, fig_w2 = st.columns(2)
    with fig_w1:
        f_sfd = go.Figure()
        f_sfd.add_trace(go.Scatter(
            x=wall_res["V"], y=wall_res["x"],
            mode="lines", fill="tozerox",
            line=dict(color="#2563EB", width=3),
            name="Shear Force (kN)"
        ))
        f_sfd.add_vline(x=0, line_dash="dash", line_color="black")
        f_sfd.update_layout(
            title="Shear Force Diagram (SFD)",
            xaxis_title="Shear Force (kN/m)",
            yaxis_title="Height above Hopper Top (m)",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(f_sfd, use_container_width=True)

    with fig_w2:
        f_bmd = go.Figure()
        f_bmd.add_trace(go.Scatter(
            x=wall_res["M"], y=wall_res["x"],
            mode="lines", fill="tozerox",
            line=dict(color="#DC2626", width=3),
            name="Bending Moment (kNm)"
        ))
        f_bmd.add_vline(x=0, line_dash="dash", line_color="black")
        f_bmd.update_layout(
            title="Bending Moment Diagram (BMD)",
            xaxis_title="Bending Moment (kNm/m)",
            yaxis_title="Height above Hopper Top (m)",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(f_bmd, use_container_width=True)

# =============================================================================
# TAB 3: HOPPER SLAB ANALYSIS
# =============================================================================
with tab3:
    st.subheader("Hopper Sloping Panel Structural Analysis")

    hop_df = pd.DataFrame([
        {
            "Hopper Face": a["Face"],
            "Slope Angle (deg)": f"{a['Angle']:.1f}°",
            "Sloping Length (m)": f"{a['Slope length']:.3f}",
            "Top Pressure (kN/m²)": f"{a['Top pressure']:.2f}",
            "Bottom Pressure (kN/m²)": f"{a['Bottom pressure']:.2f}",
            "Top Reaction (kN/m)": f"{a['Top reaction']:.2f}",
            "Outlet Reaction (kN/m)": f"{a['Outlet reaction']:.2f}",
            "Max Shear (kN/m)": f"{a['Max shear']:.2f}",
            "Max Moment (kNm/m)": f"{a['Max moment']:.2f}"
        } for a in hopper_actions
    ])
    st.dataframe(hop_df, use_container_width=True, hide_index=True)

    st.write("#### Action Diagrams along Hopper Slope")
    sel_face = st.selectbox("Select Face to View SFD & BMD:", [a["Face"] for a in hopper_actions])
    face_data = next(a for a in hopper_actions if a["Face"] == sel_face)

    col_h_sfd, col_h_bmd = st.columns(2)
    with col_h_sfd:
        f_hsfd = go.Figure()
        f_hsfd.add_trace(go.Scatter(
            x=face_data["s"], y=face_data["V"],
            mode="lines", fill="tozeroy",
            line=dict(color="#10B981", width=3),
            name="Shear Force (kN/m)"
        ))
        f_hsfd.add_hline(y=0, line_dash="dash", line_color="black")
        f_hsfd.update_layout(
            title=f"Hopper SFD – {sel_face}",
            xaxis_title="Distance along inclined face (m)",
            yaxis_title="Shear Force (kN/m)",
            template="plotly_white",
            height=380
        )
        st.plotly_chart(f_hsfd, use_container_width=True)

    with col_h_bmd:
        f_hbmd = go.Figure()
        f_hbmd.add_trace(go.Scatter(
            x=face_data["s"], y=face_data["M"],
            mode="lines", fill="tozeroy",
            line=dict(color="#F59E0B", width=3),
            name="Bending Moment (kNm/m)"
        ))
        f_hbmd.add_hline(y=0, line_dash="dash", line_color="black")
        f_hbmd.update_layout(
            title=f"Hopper BMD – {sel_face}",
            xaxis_title="Distance along inclined face (m)",
            yaxis_title="Bending Moment (kNm/m)",
            template="plotly_white",
            height=380
        )
        st.plotly_chart(f_hbmd, use_container_width=True)

# =============================================================================
# TAB 4: OUTLET BEAMS & COLUMNS
# =============================================================================
with tab4:
    st.subheader("Bottom Outlet Supporting Beams & Corner Columns")

    bw_m = outlet_beam_b / 1000.0
    bD_m = outlet_beam_D / 1000.0
    cw_m = column_b / 1000.0
    cD_m = column_D / 1000.0
    beam_sw = inp["gamma_concrete"] * bw_m * bD_m

    beam_definitions = [
        ("Longitudinal Beam 1", inp["L_outlet"], hopper_actions[0]["Outlet reaction"]),
        ("Longitudinal Beam 2", inp["L_outlet"], hopper_actions[1]["Outlet reaction"]),
        ("Transverse Beam 1", inp["W_outlet"], hopper_actions[2]["Outlet reaction"]),
        ("Transverse Beam 2", inp["W_outlet"], hopper_actions[3]["Outlet reaction"])
    ]

    outlet_beam_results = []
    for name, span, q_react in beam_definitions:
        total_unfactored_w = q_react + beam_sw
        w_u = load_factor * total_unfactored_w
        V_u = w_u * span / 2.0
        M_u = w_u * span**2 / 8.0
        outlet_beam_results.append({
            "Beam": name,
            "Span (m)": span,
            "Hopper Load (kN/m)": q_react,
            "Self Weight (kN/m)": beam_sw,
            "Factored UDL wu (kN/m)": w_u,
            "Design Shear Vu (kN)": V_u,
            "Design Moment Mu (kNm)": M_u
        })

    st.write("##### 1. Outlet Ring Beams (Factored Actions)")
    st.dataframe(pd.DataFrame(outlet_beam_results).round(2), use_container_width=True, hide_index=True)

    # Corner Columns Analysis
    # Each column supports one longitudinal and one transverse beam reaction
    col_sw = inp["gamma_concrete"] * cw_m * cD_m * column_h
    V_long = outlet_beam_results[0]["Design Shear Vu (kN)"]
    V_trans = outlet_beam_results[2]["Design Shear Vu (kN)"]
    P_u_col = V_long + V_trans + load_factor * col_sw

    st.write("##### 2. Corner Columns (Factored Axial Load Pu)")
    c_col1, c_col2, c_col3 = st.columns(3)
    c_col1.metric("Longitudinal Beam Reaction", f"{V_long:.2f} kN")
    c_col2.metric("Transverse Beam Reaction", f"{V_trans:.2f} kN")
    c_col3.metric("Total Factored Pu per Column", f"{P_u_col:.2f} kN")

# =============================================================================
# TAB 5: IS 456 RCC DESIGN & CODE VERIFICATION
# =============================================================================
with tab5:
    st.subheader("IS 456:2000 Structural Design & Compliance Checks")

    # Effective depths
    slab_th_mm = inp["t_hopper"] * 1000.0
    slab_d = slab_th_mm - cover_mm - slab_main_dia / 2.0
    beam_d = outlet_beam_D - cover_mm - beam_stirrup_dia - beam_main_dia / 2.0

    # 1. Slab Flexure Design
    slab_Mu = max(a["Max moment"] * load_factor for a in hopper_actions)
    Ast_slab_req = required_flexural_steel(slab_Mu, 1000, slab_d, inp["fck"], inp["fy"])
    Ast_slab_min = 0.0012 * 1000 * slab_th_mm
    Ast_slab_req = max(Ast_slab_req, Ast_slab_min)

    slab_main_spacing_limit = min(1000 * bar_area(slab_main_dia) / Ast_slab_req, 3 * slab_d, 300)
    Ast_slab_main_prov = 1000 * bar_area(slab_main_dia) / slab_main_spacing_limit
    slab_dist_spacing_limit = min(1000 * bar_area(slab_dist_dia) / Ast_slab_min, 5 * slab_d, 450)
    Ast_slab_dist_prov = 1000 * bar_area(slab_dist_dia) / slab_dist_spacing_limit

    # 2. Beam Design & Shear Stress Checks
    beam_reinforcement_results = []
    checks = []

    for b_item in outlet_beam_results:
        Mu_b = b_item["Design Moment Mu (kNm)"]
        Ast_b_req = required_flexural_steel(Mu_b, outlet_beam_b, beam_d, inp["fck"], inp["fy"])
        Ast_b_min = 0.85 * outlet_beam_b * beam_d / inp["fy"]
        Ast_b_req = max(Ast_b_req, Ast_b_min)

        n_bars = max(2, int(np.ceil(Ast_b_req / bar_area(beam_main_dia))))
        Ast_b_prov = n_bars * bar_area(beam_main_dia)

        xu = 0.87 * inp["fy"] * Ast_b_prov / (0.36 * inp["fck"] * outlet_beam_b)
        xu_max = limiting_steel_ratio(inp["fy"]) * beam_d
        Mu_cap = 0.87 * inp["fy"] * Ast_b_prov * (beam_d - 0.42 * xu) / 1e6

        # Shear stress check
        Vu_b = b_item["Design Shear Vu (kN)"]
        tau_v = (Vu_b * 1000.0) / (outlet_beam_b * beam_d)
        pt = 100.0 * Ast_b_prov / (outlet_beam_b * beam_d)
        tau_c = get_tau_c(pt, inp["fck"])
        tau_c_max = get_tau_c_max(inp["fck"])

        beam_reinforcement_results.append({
            "Beam": b_item["Beam"],
            "Mu (kNm)": round(Mu_b, 2),
            "Ast Req (mm²)": round(Ast_b_req, 1),
            "Bars Provided": f"{n_bars} - #{beam_main_dia} mm",
            "Ast Prov (mm²)": round(Ast_b_prov, 1),
            "Mu Capacity (kNm)": round(Mu_cap, 2),
            "xu / xu,max (mm)": f"{xu:.1f} / {xu_max:.1f}",
            "τv (MPa)": round(tau_v, 3),
            "τc (MPa)": round(tau_c, 3),
            "τc,max (MPa)": round(tau_c_max, 3)
        })

        # Add checks
        checks.append({
            "Component": b_item["Beam"],
            "Check Item": "Flexural Capacity (Mu ≤ Mu_cap)",
            "Status": "PASS" if Mu_cap >= Mu_b and xu <= xu_max else "FAIL",
            "Details": f"Demand: {Mu_b:.2f} kNm | Capacity: {Mu_cap:.2f} kNm"
        })
        checks.append({
            "Component": b_item["Beam"],
            "Check Item": "Shear Stress (τv ≤ τc,max)",
            "Status": "PASS" if tau_v <= tau_c_max else "FAIL",
            "Details": f"τv = {tau_v:.2f} MPa ≤ {tau_c_max:.2f} MPa"
        })

    # Slab Checks
    checks.insert(0, {
        "Component": "Hopper Slab",
        "Check Item": "Main Flexural Reinforcement",
        "Status": "PASS" if Ast_slab_main_prov >= Ast_slab_req else "FAIL",
        "Details": f"Req: {Ast_slab_req:.1f} mm²/m | Prov: {Ast_slab_main_prov:.1f} mm²/m"
    })
    checks.insert(1, {
        "Component": "Hopper Slab",
        "Check Item": "Distribution Steel",
        "Status": "PASS" if Ast_slab_dist_prov >= Ast_slab_min else "FAIL",
        "Details": f"Req: {Ast_slab_min:.1f} mm²/m | Prov: {Ast_slab_dist_prov:.1f} mm²/m"
    })

    # Column Axial Capacity Check
    Asc_col = col_bar_count * bar_area(col_bar_dia)
    Ac_col = column_b * column_D - Asc_col
    Pu_cap = (0.4 * inp["fck"] * Ac_col + 0.67 * inp["fy"] * Asc_col) / 1000.0
    checks.append({
        "Component": "Corner Columns",
        "Check Item": "Axial Compression Capacity (Pu ≤ Pu_cap)",
        "Status": "PASS" if Pu_cap >= P_u_col else "FAIL",
        "Details": f"Demand: {P_u_col:.1f} kN | Capacity: {Pu_cap:.1f} kN"
    })

    st.write("##### 1. Outlet Beams RCC Flexure & Shear Summary")
    st.dataframe(pd.DataFrame(beam_reinforcement_results), use_container_width=True, hide_index=True)

    st.write("##### 2. IS 456 Code Compliance Verification Checklist")
    chk_rows = []
    for c in checks:
        badge = f"<span class='badge-pass'>{c['Status']}</span>" if c["Status"] == "PASS" else f"<span class='badge-fail'>{c['Status']}</span>"
        chk_rows.append({
            "Component": c["Component"],
            "Verification Check": c["Check Item"],
            "Result": badge,
            "Details / Margin": c["Details"]
        })
    st.write(pd.DataFrame(chk_rows).to_html(escape=False, index=False), unsafe_allow_html=True)

# =============================================================================
# TAB 6: BAR BENDING SCHEDULE (BBS)
# =============================================================================
with tab6:
    st.subheader("Bar Bending Schedule (BBS) & Rebar Takeoff")

    bbs_rows = []
    mark = 1
    Li = stage1["L_internal"]
    Wi = stage1["W_internal"]
    Lo = inp["L_outlet"]
    Wo = inp["W_outlet"]
    Hh = inp["H_hopper"]

    faces = [
        {"name": "Longitudinal Face 1", "run": (Wi - Wo) / 2, "edge": Lo},
        {"name": "Longitudinal Face 2", "run": (Wi - Wo) / 2, "edge": Lo},
        {"name": "Transverse Face 1", "run": (Li - Lo) / 2, "edge": Wo},
        {"name": "Transverse Face 2", "run": (Li - Lo) / 2, "edge": Wo}
    ]

    for face in faces:
        slope_len_mm = math.sqrt(face["run"]**2 + Hh**2) * 1000
        edge_mm = face["edge"] * 1000
        # Main bars
        m_cnt = number_of_bars(edge_mm - 2 * cover_mm, slab_main_spacing_limit)
        m_cut = slope_len_mm - 2 * cover_mm + 2 * 150
        bbs_rows.append(make_bbs_row(mark, face["name"], "Slab Main Bars", slab_main_dia, m_cnt, m_cut))
        mark += 1
        # Dist bars
        d_cnt = number_of_bars(slope_len_mm - 2 * cover_mm, slab_dist_spacing_limit)
        d_cut = edge_mm - 2 * cover_mm + 2 * 150
        bbs_rows.append(make_bbs_row(mark, face["name"], "Slab Dist Bars", slab_dist_dia, d_cnt, d_cut))
        mark += 1

    # Outlet Beams
    for b in outlet_beam_results:
        b_span_mm = b["Span (m)"] * 1000
        bbs_rows.append(make_bbs_row(mark, b["Beam"], "Longitudinal Tension Bars", beam_main_dia, 3, b_span_mm + 600))
        mark += 1
        stirrup_cnt = number_of_bars(b_span_mm, beam_stirrup_spacing)
        stirrup_cut = 2 * (outlet_beam_b - 2 * cover_mm) + 2 * (outlet_beam_D - 2 * cover_mm) + 150
        bbs_rows.append(make_bbs_row(mark, b["Beam"], "Stirrups (2-legged)", beam_stirrup_dia, stirrup_cnt, stirrup_cut))
        mark += 1

    # Columns (4 columns)
    col_len_mm = column_h * 1000
    bbs_rows.append(make_bbs_row(mark, "4 Corner Columns", "Column Main Bars", col_bar_dia, 4 * col_bar_count, col_len_mm + 600))
    mark += 1
    ties_per_col = number_of_bars(col_len_mm, col_tie_spacing)
    tie_cut = 2 * (column_b - 2 * cover_mm) + 2 * (column_D - 2 * cover_mm) + 150
    bbs_rows.append(make_bbs_row(mark, "4 Corner Columns", "Lateral Ties", col_tie_dia, 4 * ties_per_col, tie_cut))
    mark += 1

    df_bbs = pd.DataFrame(bbs_rows)
    st.dataframe(df_bbs, use_container_width=True, hide_index=True)

    # Steel Weight Summary by Diameter
    dia_summary = df_bbs.groupby("Diameter (mm)")["Total weight (kg)"].sum().reset_index()
    total_rebar_weight = dia_summary["Total weight (kg)"].sum()

    st.write("##### Steel Takeoff Summary by Rebar Diameter")
    c_bbs1, c_bbs2 = st.columns([1, 1])
    with c_bbs1:
        st.dataframe(dia_summary.round(2), use_container_width=True, hide_index=True)
    with c_bbs2:
        st.metric("Total Steel Rebar Required", f"{total_rebar_weight / 1000.0:.2f} Tonnes", f"{total_rebar_weight:.1f} kg")

    # CSV Download
    csv_data = df_bbs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Bar Bending Schedule (CSV)",
        data=csv_data,
        file_name="RCC_Bunker_BBS.csv",
        mime="text/csv"
    )

# =============================================================================
# TAB 7: DRAWINGS (2D & 3D)
# =============================================================================
with tab7:
    st.subheader("Structural Drawings & Reinforcement Detailing")

    sub_draw_tab1, sub_draw_tab2, sub_draw_tab3 = st.tabs([
        "📐 2D Plan View",
        "📏 2D Section View",
        "🕸️ 3D Reinforcement Cage"
    ])

    with sub_draw_tab1:
        fig_plan, ax_plan = plt.subplots(figsize=(10, 8))
        L = inp["L_outer"]
        W = inp["W_outer"]
        Li = stage1["L_internal"]
        Wi = stage1["W_internal"]
        Lo = inp["L_outlet"]
        Wo = inp["W_outlet"]

        # Outer & Inner Perimeter
        ax_plan.plot([-L/2, L/2, L/2, -L/2, -L/2], [-W/2, -W/2, W/2, W/2, -W/2], "k-", lw=3, label="Outer Wall")
        ax_plan.plot([-Li/2, Li/2, Li/2, -Li/2, -Li/2], [-Wi/2, -Wi/2, Wi/2, Wi/2, -Wi/2], "k--", lw=1.5, label="Inner Wall")
        ax_plan.plot([-Lo/2, Lo/2, Lo/2, -Lo/2, -Lo/2], [-Wo/2, -Wo/2, Wo/2, Wo/2, -Wo/2], "b-", lw=2, label="Outlet Opening")

        # Corner columns
        for cx in [-L/2, L/2]:
            for cy in [-W/2, W/2]:
                ax_plan.add_patch(plt.Rectangle((cx - 0.15, cy - 0.15), 0.3, 0.3, color="gray", alpha=0.5))

        # Hip lines from corners to outlet
        ax_plan.plot([-Li/2, -Lo/2], [-Wi/2, -Wo/2], "r-", lw=2, label="Hip Corner Rebar")
        ax_plan.plot([Li/2, Lo/2], [-Wi/2, -Wo/2], "r-", lw=2)
        ax_plan.plot([Li/2, Lo/2], [Wi/2, Wo/2], "r-", lw=2)
        ax_plan.plot([-Li/2, -Lo/2], [Wi/2, Wo/2], "r-", lw=2)

        ax_plan.set_title("BUNKER REINFORCEMENT – 2D PLAN VIEW", fontsize=14, fontweight="bold")
        ax_plan.set_xlabel("Length (m)")
        ax_plan.set_ylabel("Width (m)")
        ax_plan.legend(loc="upper right")
        ax_plan.set_aspect("equal")
        ax_plan.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig_plan)

    with sub_draw_tab2:
        fig_sec, ax_sec = plt.subplots(figsize=(10, 7))
        Hw = inp["H_wall"]
        Hh = inp["H_hopper"]
        tw = inp["t_wall"]
        col_H = column_h

        # Wall outlines
        ax_sec.plot([-L/2, -L/2], [0, Hw], "k-", lw=4)
        ax_sec.plot([L/2, L/2], [0, Hw], "k-", lw=4)
        ax_sec.plot([-L/2, L/2], [Hw, Hw], color="purple", lw=5, label="Top Edge Beam")

        # Hopper slope
        ax_sec.plot([-L/2 + tw, -Lo/2], [0, -Hh], "k-", lw=4)
        ax_sec.plot([L/2 - tw, Lo/2], [0, -Hh], "k-", lw=4)
        ax_sec.plot([-Lo/2, Lo/2], [-Hh, -Hh], color="darkorange", lw=6, label="Outlet Beam")

        # Columns
        for cx in [-L/2, L/2]:
            ax_sec.plot([cx, cx], [-Hh - col_H, 0], "k-", lw=6)
            ax_sec.plot([cx - 0.1, cx - 0.1], [-Hh - col_H, 0], "r-", lw=1.5, label="Column Rebar" if cx == -L/2 else "")
            ax_sec.plot([cx + 0.1, cx + 0.1], [-Hh - col_H, 0], "r-", lw=1.5)

        ax_sec.set_title("BUNKER REINFORCEMENT – 2D SECTION VIEW", fontsize=14, fontweight="bold")
        ax_sec.set_xlabel("Length (m)")
        ax_sec.set_ylabel("Elevation (m)")
        ax_sec.grid(True, linestyle=":", alpha=0.6)
        ax_sec.legend(loc="upper right")
        ax_sec.set_aspect("equal")
        st.pyplot(fig_sec)

    with sub_draw_tab3:
        # Full 3D interactive wireframe of reinforcement cage
        fig_3d_rebar = go.Figure()

        def add_line_3d(points, color="red", width=4, name=None):
            fig_3d_rebar.add_trace(go.Scatter3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                mode="lines",
                line=dict(color=color, width=width),
                name=name,
                showlegend=(name is not None)
            ))

        # Top edge beams
        top_z = inp["H_wall"]
        add_line_3d([(-L/2, -W/2, top_z), (L/2, -W/2, top_z), (L/2, W/2, top_z), (-L/2, W/2, top_z), (-L/2, -W/2, top_z)], color="purple", width=6, name="Top Edge Beam")

        # Outlet beams
        out_z = -Hh
        add_line_3d([(-Lo/2, -Wo/2, out_z), (Lo/2, -Wo/2, out_z), (Lo/2, Wo/2, out_z), (-Lo/2, Wo/2, out_z), (-Lo/2, -Wo/2, out_z)], color="orange", width=6, name="Outlet Ring Beam")

        # Corner columns
        bot_z = -Hh - column_h
        for i, (cx, cy) in enumerate([(-L/2, -W/2), (L/2, -W/2), (L/2, W/2), (-L/2, W/2)]):
            for angle in np.linspace(0, 2*np.pi, col_bar_count, endpoint=False):
                bx = cx + 0.1 * np.cos(angle)
                by = cy + 0.1 * np.sin(angle)
                add_line_3d([(bx, by, bot_z), (bx, by, 0)], color="red", width=3, name="Column Main Bar" if i == 0 and angle == 0 else None)

        fig_3d_rebar.update_layout(
            title="3D Reinforcement Cage Model",
            scene=dict(
                xaxis_title="Length (m)",
                yaxis_title="Width (m)",
                zaxis_title="Elevation (m)",
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=550
        )
        st.plotly_chart(fig_3d_rebar, use_container_width=True)

# =============================================================================
# TAB 8: REPORT GENERATION & PDF EXPORT
# =============================================================================
with tab8:
    st.subheader("Calculation Report & Executive Summary")

    st.markdown(f"""
    ### Executive Design Summary
    - **Project:** Rectangular RCC Hopper Bunker Design
    - **Stored Bulk Material:** {inp['material_name']} ($\gamma = {inp['gamma_material']}$ kN/m³)
    - **Concrete & Steel:** Grade M{inp['fck']} Concrete, Fe {inp['fy']} Reinforcement
    - **Volumetric Capacity:** {stage1['V_capacity']:.2f} m³ ({stage1['W_material']:.1f} kN of stored material)
    - **Total Gravity Action on Substructure:** {stage1['W_total_simple']:.1f} kN
    - **Governing Outlet Beam Moment:** {max(b['Design Moment Mu (kNm)'] for b in outlet_beam_results):.2f} kNm
    - **Corner Column Design Axial Load (Pu):** {P_u_col:.1f} kN
    - **Total Rebar Quantity:** {total_rebar_weight / 1000.0:.2f} Tonnes ({total_rebar_weight:.1f} kg)
    """)

    if HAS_REPORTLAB:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=15*mm, leftMargin=15*mm,
            topMargin=15*mm, bottomMargin=15*mm
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Title"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1E3A8A"),
            alignment=1
        )
        head_style = ParagraphStyle(
            name="HeadingStyle",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1E3A8A"),
            spaceBefore=10,
            spaceAfter=6
        )

        story = [
            Paragraph("RECTANGULAR RCC HOPPER BUNKER", title_style),
            Paragraph("Structural Design & Code Verification Report (IS 4995 & IS 456)", styles["Normal"]),
            Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("1. DESIGN BASIS & GEOMETRY", head_style)
        ]

        # Geometry Table
        t_geo_data = [
            ["Parameter", "Design Value"],
            ["Bunker Outer Plan", f"{inp['L_outer']} m × {inp['W_outer']} m"],
            ["Internal Plan", f"{stage1['L_internal']:.2f} m × {stage1['W_internal']:.2f} m"],
            ["Vertical Wall Height", f"{inp['H_wall']} m (t = {inp['t_wall']} m)"],
            ["Hopper Height / Outlet", f"{inp['H_hopper']} m / ({inp['L_outlet']} m × {inp['W_outlet']} m)"],
            ["Material Unit Weight", f"{inp['gamma_material']} kN/m³ ({inp['material_name']})"],
            ["Total Capacity", f"{stage1['V_capacity']:.2f} m³ ({stage1['W_material']:.1f} kN)"],
            ["Concrete / Steel Grade", f"M{inp['fck']} / Fe {inp['fy']}"]
        ]
        t_geo = Table(t_geo_data, colWidths=[80*mm, 90*mm])
        t_geo.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")])
        ]))
        story.append(t_geo)
        story.append(Spacer(1, 10))

        story.append(Paragraph("2. STRUCTURAL VERIFICATION CHECKLIST (IS 456:2000)", head_style))
        t_chk_data = [["Component", "Check Description", "Result", "Details"]]
        for c in checks:
            t_chk_data.append([c["Component"], c["Check Item"], c["Status"], c["Details"]])

        t_chk = Table(t_chk_data, colWidths=[40*mm, 55*mm, 20*mm, 55*mm])
        t_chk.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])
        ]))
        story.append(t_chk)

        doc.build(story)
        pdf_bytes = pdf_buffer.getvalue()

        st.download_button(
            label="📄 Download Complete PDF Design Report",
            data=pdf_bytes,
            file_name="RCC_Hopper_Bunker_Design_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Install 'reportlab' to enable automatic PDF design report download.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "RCC Hopper Bunker Design Software • Framework: Indian Standards (IS 4995 & IS 456) • Starter Engineering Tool"
    "</div>",
    unsafe_allow_html=True
)
