import streamlit as st
import pyvista as pv
import numpy as np
import os
import sys

import tempfile
import base64
from datetime import datetime
import streamlit.components.v1 as components

# Import core engine modules
from config import CONFIG, get_frequency
from core.implicit_base import Sphere
from core.mesh_container import MeshSDF
from core.analytics import calculate_volume_fraction, gibson_ashby_stiffness, estimate_mass
from lattices.tpms import Gyroid, Diamond, HybridLattice, Intersection
from lattices.graded_lattice import GradedLattice, linear_z_grading, point_attractor_grading
from export.stl_exporter import save_mesh_to_stl

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="ImplicitLattice | Engine", layout="wide", initial_sidebar_state="expanded")

# --- SLEEK MONOCHROME THEME INJECTION ---
st.markdown("""
    <style>
    /* 1. CONTAINER BLACKOUT */
    :root {
        --primary-color: #00E5FF !important;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }

    /* Target the vertical blocks and their direct children to ensure black background between widgets */
    [data-testid="stVerticalBlock"] > div {
        background-color: #000000 !important;
    }

    /* 2. REVERTED NAVIGATION (SAFELY ENFORCING TYPOGRAPHY) */
    .stMarkdown, p, label, .stMetricValue {
        font-family: 'Helvetica', 'Arial', sans-serif !important;
    }

    /* Target main text containers but avoid interface buttons/icons */
    [data-testid="stAppViewContainer"] {
        font-family: 'Helvetica', 'Arial', sans-serif !important;
    }

    /* 3. SLIDER VISIBILITY (RESTORATION) */
    /* The track line */
    div[data-baseweb="slider"] > div:first-child > div:first-child {
        background: #00E5FF !important;
    }
    /* The secondary track (unused part) */
    div[data-baseweb="slider"] > div:first-child {
        background-color: #222222 !important;
    }
    /* The handle (thumb) */
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #00E5FF !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.8) !important;
    }
    /* Slider value text */
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        color: #00E5FF !important;
        font-weight: 600 !important;
    }

    /* 4. NUMERIC INPUT (+/-) */
    button[data-testid="stNumericInputStepUp"], 
    button[data-testid="stNumericInputStepDown"] {
        background-color: transparent !important;
        color: #00E5FF !important;
        border: 1px solid #333333 !important;
    }
    button[data-testid="stNumericInputStepUp"]:hover, 
    button[data-testid="stNumericInputStepDown"]:hover {
        border-color: #00E5FF !important;
        background-color: rgba(0, 229, 255, 0.1) !important;
    }
    button[data-testid="stNumericInputStepUp"] svg, 
    button[data-testid="stNumericInputStepDown"] svg {
        fill: #00E5FF !important;
    }

    /* Input Box */
    div[data-baseweb="input"] {
        background-color: #080808 !important;
        border: 1px solid #222 !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #00E5FF !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.3) !important;
    }
    input {
        color: #00E5FF !important;
    }

    /* 5. PRIMARY GENERATE BUTTON */
    div.stButton > button:first-child {
        background-color: #000000 !important;
        color: #00E5FF !important;
        border: 1px solid #00E5FF !important;
        border-radius: 2px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.1);
        width: 100%;
        height: 3em;
    }
    div.stButton > button:first-child:hover {
        background-color: rgba(0, 229, 255, 0.1) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
        color: #FFFFFF !important;
    }

    /* 6. METRICS & STATUS */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        border-bottom: 1px solid #00E5FF;
        box-shadow: 0 4px 10px -5px rgba(0, 229, 255, 0.8);
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
        font-size: 0.7rem !important;
    }

    div[data-testid="stStatusWidget"] {
        border: 1px solid #00E5FF !important;
        background-color: #000000 !important;
    }
    div[data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p {
        color: #00E5FF !important;
        font-family: monospace;
    }

    /* 7. UTILITIES & CLEANUP */
    hr { border-top: 1px solid #1A1A1A !important; }
    #MainMenu, footer { visibility: hidden; }
    header { visibility: visible !important; background: rgba(0,0,0,0.5) !important; }
    
    /* Ensure the sidebar toggle is ALWAYS visible and Cyber Blue */
    button[data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block !important;
        color: #00E5FF !important;
    }
    
    /* Sidebar Border */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #222 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("IMPLICIT LATTICE")
st.caption("DIGITAL TWIN SYNTHESIS ENGINE | v2.0")

# --- SIDEBAR: DESIGN INPUTS ---
st.sidebar.markdown("### CONFIGURATION")

# Container Upload (Moved up for scope visibility)
st.sidebar.markdown("#### BOUNDARY MESH")
uploaded_file = st.sidebar.file_uploader("UPLOAD STL", type=["stl"], label_visibility="collapsed")

st.sidebar.markdown("---")

# Selection inputs
st.sidebar.markdown("#### ARCHITECTURE")
hybrid_mode = st.sidebar.toggle("HYBRID (GYROID/DIAMOND)", value=False)
if hybrid_mode:
    blend_weight = st.sidebar.slider("BLEND WEIGHT", 0.0, 1.0, 0.5)
    lattice_type = "Hybrid"
else:
    lattice_type = st.sidebar.selectbox("LATTICE TYPE", ["Gyroid", "Diamond"])
cell_size = st.sidebar.slider("CELL SIZE (MM)", 0.2, 5.0, 0.8)
resolution = st.sidebar.slider("RESOLUTION", 20, 150, 60)
slice_model = st.sidebar.toggle("INTERNAL SECTION (X-Z)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("#### GRADING")
grading_mode = st.sidebar.selectbox("STRATEGY", ["Linear Z-Grading", "Point Attractor (Reinforcement)"])
t_min = st.sidebar.number_input("MIN THICKNESS", 0.05, 1.0, 0.1)
t_max = st.sidebar.number_input("MAX THICKNESS", 0.05, 1.0, 0.5)

# Attractor Point State
if "att_x" not in st.session_state: st.session_state.att_x = 0.0
if "att_y" not in st.session_state: st.session_state.att_y = 0.0
if "att_z" not in st.session_state: st.session_state.att_z = 0.0

if grading_mode == "Point Attractor (Reinforcement)":
    st.sidebar.markdown("#### ATTRACTOR COORDINATES")
    col1, col2, col3 = st.sidebar.columns(3)
    ax = col1.number_input("X", value=st.session_state.att_x, format="%.2f", key="input_ax")
    ay = col2.number_input("Y", value=st.session_state.att_y, format="%.2f", key="input_ay")
    az = col3.number_input("Z", value=st.session_state.att_z, format="%.2f", key="input_az")
    attractor_radius = st.sidebar.slider("INFLUENCE RADIUS", 0.1, 20.0, 5.0)
    
    if st.sidebar.button("CALC CENTER"):
        # Auto-calculate center from container
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            temp_mesh = pv.read(tmp_path)
            b = temp_mesh.bounds
            st.session_state.att_x = (b[0] + b[1]) / 2
            st.session_state.att_y = (b[2] + b[3]) / 2
            st.session_state.att_z = (b[4] + b[5]) / 2
            os.remove(tmp_path)
            st.rerun()
        else:
            st.session_state.att_x = 0.0
            st.session_state.att_y = 0.0
            st.session_state.att_z = 0.0
            st.rerun()
else:
    # Set defaults for Linear Z if needed, though they aren't used in that mode's UI
    ax, ay, az, attractor_radius = 0.0, 0.0, 0.0, 0.0

st.sidebar.markdown("---")

# --- SESSION STATE FOR PERSISTENCE ---
if "final_stl_path" not in st.session_state:
    st.session_state.final_stl_path = None
if "vol_frac" not in st.session_state:
    st.session_state.vol_frac = 0.0
if "mass" not in st.session_state:
    st.session_state.mass = 0.0

# --- MAIN GENERATION LOGIC ---
if st.sidebar.button("GENERATE STRUCTURE"):
    with st.status("ENGINEERING LATTICE...", expanded=True) as status:
        try:
            # 1. Handle Container
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    container_path = tmp_file.name
                st.write(f"Voxelizing uploaded container...")
                container = MeshSDF(container_path, use_cache=True, cache_resolution=50)
            else:
                st.write("No STL uploaded. Using default Sphere container.")
                container = Sphere(radius=1.2)

            # 2. Lattice Synthesis
            b = container.bounds
            freq = (2 * np.pi) / cell_size
            
            if hybrid_mode:
                l1 = Gyroid(frequency=freq)
                l2 = Diamond(frequency=freq)
                base_lattice = HybridLattice(l1, l2, weight=blend_weight)
                st.write(f"Synthesizing Hybrid Meta-Material (w={blend_weight:.2f})...")
            else:
                if lattice_type == "Gyroid":
                    base_lattice = Gyroid(frequency=freq)
                else:
                    base_lattice = Diamond(frequency=freq)
                st.write(f"Synthesizing {lattice_type} Architecture...")
            
            # Apply Grading Strategy
            if grading_mode == "Point Attractor (Reinforcement)":
                st.write(f"Applying Reinforcement Point at ({ax:.2f}, {ay:.2f}, {az:.2f})...")
                # Wrap it in a function that GradedLattice expects (x, y, z)
                grading = lambda x, y, z: point_attractor_grading((x, y, z), (ax, ay, az), attractor_radius, t_min, t_max)
            else:
                st.write(f"Applying Linear Z-Grading from {b[4]:.2f} to {b[5]:.2f}...")
                grading = linear_z_grading(z_min=b[4], z_max=b[5], t_min=t_min, t_max=t_max)
            
            graded = GradedLattice(base_lattice, grading)
            final_part = Intersection(container, graded)

            # 3. Analytics
            st.write("Performing Engineering Analytics...")
            st.session_state.vol_frac = calculate_volume_fraction(final_part, b, resolution=40)
            total_vol = (b[1]-b[0]) * (b[3]-b[2]) * (b[5]-b[4])
            st.session_state.mass = estimate_mass(st.session_state.vol_frac, total_vol, 2.7)

            # 4. Mesh Generation
            st.write(f"Generating 3D Mesh at resolution {resolution}...")
            pv.OFF_SCREEN = True
            plotter = pv.Plotter()
            
            # Bounding box slightly larger than container
            render_bounds = (b[0]-0.05, b[1]+0.05, b[2]-0.05, b[3]+0.05, b[4]-0.05, b[5]+0.05)
            
            grid = pv.ImageData(
                dimensions=(resolution, resolution, resolution),
                spacing=((render_bounds[1]-render_bounds[0])/(resolution-1),
                         (render_bounds[3]-render_bounds[2])/(resolution-1),
                         (render_bounds[5]-render_bounds[4])/(resolution-1)),
                origin=(render_bounds[0], render_bounds[2], render_bounds[4])
            )
            
            points = grid.points
            values = final_part.evaluate(points[:,0], points[:,1], points[:,2])
            grid.point_data["values"] = values
            mesh = grid.contour([0.0])

            # Validation Check
            if mesh.n_points == 0:
                st.error("Lattice out of bounds or too thin to resolve. Try decreasing thickness or increasing resolution.")
                st.stop()

            # --- COLOR BY THICKNESS ---
            # Evaluate thickness function at mesh vertices
            # Note: 'grading' is a lambda that takes (x, y, z)
            m_pts = mesh.points
            thickness_scalars = grading(m_pts[:,0], m_pts[:,1], m_pts[:,2])
            
            # Normalize for colormap (t_min to t_max)
            t_range = t_max - t_min if t_max > t_min else 1.0
            norm_thickness = np.clip((thickness_scalars - t_min) / t_range, 0, 1)
            
            # Create a simple 'Viridis-like' colormap (Purple to Yellow)
            # We'll use this to color the PyVista plotter and pass to Three.js
            mesh.point_data["thickness"] = thickness_scalars

            # --- CLIPPING ---
            if slice_model:
                st.write("Slicing model for internal inspection...")
                mesh = mesh.clip(normal='y', origin=(0, (b[2]+b[3])/2, 0))

            # 5. Export for Viewer (Custom Robust Three.js Implementation)
            st.write("Finalizing 3D Scene...")
            VIEWER_PATH = os.path.abspath("temp_viewer.html")
            FALLBACK_PATH = os.path.abspath("fallback_preview.png")
            
            # Save fallback screenshot
            plotter.set_background("#000000")
            # Reverting to 'plasma' for the 'old' professional gradient look
            plotter.add_mesh(mesh, scalars="thickness", cmap="plasma", show_edges=False)
            plotter.reset_camera()
            plotter.screenshot(FALLBACK_PATH)
            plotter.close()

            # Generate STL for the viewer
            temp_stl = "temp_view_data.stl"
            save_mesh_to_stl(mesh, temp_stl)
            
            import base64
            with open(temp_stl, "rb") as f:
                stl_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Load externalized shaders and engine JS
            base_dir = os.path.dirname(os.path.abspath(__file__))
            vertex_path = os.path.join(base_dir, "src", "shaders", "vertex.glsl")
            fragment_path = os.path.join(base_dir, "src", "shaders", "fragment.glsl")
            engine_path = os.path.join(base_dir, "src", "core", "engine.js")
            
            try:
                with open(vertex_path, "r", encoding="utf-8") as f:
                    vertex_shader_code = f.read()
                with open(fragment_path, "r", encoding="utf-8") as f:
                    fragment_shader_code = f.read()
                with open(engine_path, "r", encoding="utf-8") as f:
                    engine_js_template = f.read()
            except Exception as read_err:
                st.error(f"Failed to load WebGL source assets from src/: {str(read_err)}")
                st.stop()

            # Map Python parameters and shaders directly to the JavaScript WebGL engine
            engine_js = (engine_js_template
                .replace("__STL_BASE64__", stl_base64)
                .replace("__T_MIN__", f"{t_min:.4f}")
                .replace("__T_MAX__", f"{t_max:.4f}")
                .replace("__GRADING_MODE__", "1" if grading_mode == "Point Attractor (Reinforcement)" else "0")
                .replace("__ATT_X__", f"{ax:.4f}")
                .replace("__ATT_Y__", f"{ay:.4f}")
                .replace("__ATT_Z__", f"{az:.4f}")
                .replace("__ATT_RAD__", f"{attractor_radius:.4f}")
                .replace("__Z_MIN__", f"{b[4]:.4f}")
                .replace("__Z_MAX__", f"{b[5]:.4f}")
                .replace("__VERTEX_SHADER__", vertex_shader_code)
                .replace("__FRAGMENT_SHADER__", fragment_shader_code)
            )

            # Create a professional Standalone Three.js Viewer with Vertex Coloring
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ImplicitLattice Viewer</title>
                <style>
                    body {{ margin: 0; background-color: #000000; overflow: hidden; font-family: 'Helvetica', sans-serif; }}
                    canvas {{ width: 100%; height: 100%; }}
                    #info {{ position: absolute; top: 15px; width: 100%; text-align: center; color: #444; font-size: 10px; text-transform: uppercase; letter-spacing: 2px; pointer-events: none; }}
                    #legend {{ position: absolute; bottom: 20px; right: 20px; background: rgba(0,0,0,0.8); padding: 12px; border-radius: 2px; color: #888; border: 1px solid #222; }}
                </style>
            </head>
            <body>
                <div id="info">Digital Twin Simulation Data | Real-time 3D Engine</div>
                <div id="legend">
                    <div style="font-size: 8px; margin-bottom: 5px; text-transform: uppercase;">Lattice Thickness (mm)</div>
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 10px; margin-right: 8px;">{t_min:.2f}</span>
                        <div style="width: 120px; height: 4px; background: linear-gradient(to right, #0d0887, #9c179e, #ed7953, #f0f921);"></div>
                        <span style="font-size: 10px; margin-left: 8px;">{t_max:.2f}</span>
                    </div>
                </div>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
                <script>
                    {engine_js}
                </script>
            </body>
            </html>
            """
            
            with open(VIEWER_PATH, "w", encoding="utf-8") as f:
                f.write(html_template)
            
            if os.path.exists(temp_stl): os.remove(temp_stl)

            # 6. Prepare Download
            st.session_state.final_stl_path = f"export/web_generated_{datetime.now().strftime('%H%M%S')}.stl"
            os.makedirs("export", exist_ok=True)
            save_mesh_to_stl(mesh, st.session_state.final_stl_path)
            
            status.update(label="STRUCTURE VALIDATED | READY", state="complete")
        except Exception as e:
            st.error(f"Generation Failed: {str(e)}")
            status.update(label="ERROR IN SYNTHESIS", state="error")

# --- DISPLAY RESULTS ---
col1, col2 = st.columns([3, 1])

with col1:
    VIEWER_PATH = os.path.abspath("temp_viewer.html")
    FALLBACK_PATH = os.path.abspath("fallback_preview.png")
    
    if os.path.exists(VIEWER_PATH):
        try:
            with open(VIEWER_PATH, 'r', encoding='utf-8') as f:
                html_data = f.read()
            
            # Since we generate our own HTML, we don't expect 404 anymore
            st.components.v1.html(html_data, height=700, scrolling=False)
            
        except Exception as e:
            st.error(f"Viewer Error: {str(e)}")
            if os.path.exists(FALLBACK_PATH):
                st.image(FALLBACK_PATH, caption="3D Preview (Static Fallback)")
    else:
        # Default state BEFORE any generation has happened
        st.info("Configure your lattice and click 'Generate' to visualize the 3D model.")
        
        # If an old fallback exists, we can show it as a teaser or just keep it clean
        if os.path.exists(FALLBACK_PATH):
            st.image(FALLBACK_PATH, caption="Previous Generation Preview", use_container_width=True)

with col2:
    st.markdown("#### ANALYTICS")
    st.metric("VOL FRACTION", f"{st.session_state.vol_frac:.1%}")
    st.metric("EST MASS", f"{st.session_state.mass:.2f} G")
    
    st.markdown("---")
    if st.session_state.final_stl_path and os.path.exists(st.session_state.final_stl_path):
        with open(st.session_state.final_stl_path, "rb") as file:
            st.download_button(
                label="DOWNLOAD STL",
                data=file,
                file_name=os.path.basename(st.session_state.final_stl_path),
                mime="application/sla"
            )
        
        # --- PDF REPORT GENERATION ---
        if st.button("EXPORT PDF REPORT"):
            try:
                from fpdf import FPDF
            except ImportError:
                # Last resort inline install for isolated environments
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
                from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(200, 10, text="ImplicitLattice Engineering Report", new_x="LMARGIN", new_y="NEXT", align='C')
            pdf.set_font("Helvetica", size=10)
            pdf.cell(200, 10, text=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align='C')
            pdf.ln(10)
            
            # Screenshot
            FALLBACK_PATH = os.path.abspath("fallback_preview.png")
            if os.path.exists(FALLBACK_PATH):
                pdf.image(FALLBACK_PATH, x=50, y=40, w=110)
                pdf.ln(80)
            
            # Metrics Table
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(200, 10, text="Part Specifications & Analytics", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
            pdf.cell(100, 8, text=f"Lattice Type: {lattice_type}", border=1)
            pdf.cell(90, 8, text=f"Cell Size: {cell_size} mm", border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.cell(100, 8, text=f"Grading Mode: {grading_mode}", border=1)
            pdf.cell(90, 8, text=f"Thickness: {t_min:.2f} - {t_max:.2f} mm", border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)
            
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(200, 10, text="Simulation Results", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(100, 10, text="Volume Fraction", border=1, fill=True)
            pdf.cell(90, 10, text=f"{st.session_state.vol_frac:.2%}", border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.cell(100, 10, text="Estimated Mass", border=1, fill=True)
            pdf.cell(90, 10, text=f"{st.session_state.mass:.2f} g", border=1, new_x="LMARGIN", new_y="NEXT")
            
            pdf_output = pdf.output()
            st.download_button(
                label="Click to Download PDF",
                data=bytes(pdf_output),
                file_name=f"ImplicitLattice_Report_{datetime.now().strftime('%H%M%S')}.pdf",
                mime="application/pdf"
            )

    else:
        st.write("Generate a mesh to enable export.")

st.markdown("---")
st.caption("IMPLICIT LATTICE | ARCHITECTURAL SIMULATION | 2026")
