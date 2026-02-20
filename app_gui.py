import streamlit as st
import pyvista as pv
import numpy as np
import os
import tempfile
from datetime import datetime
import streamlit.components.v1 as components

# Import core engine modules
from config import CONFIG, get_frequency
from core.implicit_base import Sphere
from core.mesh_container import MeshSDF
from core.analytics import calculate_volume_fraction, gibson_ashby_stiffness, estimate_mass
from lattices.tpms import Gyroid, Intersection # Add Diamond if available
from lattices.graded_lattice import GradedLattice, linear_z_grading
from export.stl_exporter import save_mesh_to_stl

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="ImplicitLattice Engine", layout="wide", initial_sidebar_state="expanded")

# Apply a professional dark theme style
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4b5cf2; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #28a745; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ ImplicitLattice: Function-Defined Modeling Engine")
st.subheader("Professional CAD Generation for Additive Manufacturing")

# --- SIDEBAR: DESIGN INPUTS ---
st.sidebar.header("🛠️ Lattice Configuration")

# Selection inputs
lattice_type = st.sidebar.selectbox("Lattice Architecture", ["Gyroid", "Diamond (Standard)"])
cell_size = st.sidebar.slider("Cell Size (mm)", 0.2, 5.0, 0.8)
resolution = st.sidebar.slider("Mesh Resolution", 20, 150, 60)

st.sidebar.markdown("---")
st.sidebar.header("📐 Grading Parameters")
t_min = st.sidebar.number_input("Min Thickness (t_min)", 0.05, 1.0, 0.1)
t_max = st.sidebar.number_input("Max Thickness (t_max)", 0.05, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("📁 Container STL")
uploaded_file = st.sidebar.file_uploader("Upload Boundary Mesh (STL)", type=["stl"])

# --- SESSION STATE FOR PERSISTENCE ---
if "final_stl_path" not in st.session_state:
    st.session_state.final_stl_path = None
if "vol_frac" not in st.session_state:
    st.session_state.vol_frac = 0.0
if "mass" not in st.session_state:
    st.session_state.mass = 0.0

# --- MAIN GENERATION LOGIC ---
if st.sidebar.button("🚀 Generate Structure"):
    with st.status("🛠️ Engineering Lattice...", expanded=True) as status:
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
            base_lattice = Gyroid(frequency=freq)
            
            # Dynamic Z-bounds for grading
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

            # 5. Export for Viewer
            plotter.set_background("#1e1e1e") # Dark Grey Background
            plotter.add_mesh(mesh, color="lightblue", show_edges=True, smooth_shading=True)
            
            # Studio Lighting and Eye Dome Lighting
            plotter.add_light(pv.Light(position=(2, 2, 5), intensity=1.5))
            plotter.enable_eye_dome_lighting()
            
            # Center and Zoom
            plotter.view_isometric()
            plotter.reset_camera()
            plotter.zoom_camera(0.9)
            
            html_abs_path = os.path.abspath("temp_viewer.html")
            plotter.export_html(html_abs_path)
            
            # 6. Prepare Download
            st.session_state.final_stl_path = f"export/web_generated_{datetime.now().strftime('%H%M%S')}.stl"
            os.makedirs("export", exist_ok=True)
            save_mesh_to_stl(mesh, st.session_state.final_stl_path)
            
            status.update(label="✅ Structure Validated & Ready!", state="complete")
        except Exception as e:
            st.error(f"Generation Failed: {str(e)}")
            status.update(label="❌ Error in Synthesis", state="error")

# --- DISPLAY RESULTS ---
col1, col2 = st.columns([3, 1])

with col1:
    if os.path.exists("temp_viewer.html"):
        import time
        with open("temp_viewer.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Using a direct HTML component (removed 'key' as it's not supported in this method)
        st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.info("👈 Configure your lattice and click 'Generate' to visualize the 3D model.")

with col2:
    st.subheader("📊 Part Analytics")
    st.metric("Volume Fraction", f"{st.session_state.vol_frac:.1%}")
    st.metric("Estimated Mass", f"{st.session_state.mass:.2f} g")
    
    st.markdown("---")
    st.subheader("📥 Commercial Export")
    if st.session_state.final_stl_path and os.path.exists(st.session_state.final_stl_path):
        with open(st.session_state.final_stl_path, "rb") as file:
            st.download_button(
                label="Download STL File",
                data=file,
                file_name=os.path.basename(st.session_state.final_stl_path),
                mime="application/sla"
            )
    else:
        st.write("Generate a mesh to enable export.")

st.markdown("---")
st.caption("© 2026 ImplicitLattice | Powered by Streamlit & PyVista")
