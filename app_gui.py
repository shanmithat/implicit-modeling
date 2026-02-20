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
from lattices.tpms import Gyroid, Diamond, HybridLattice, Intersection
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
hybrid_mode = st.sidebar.toggle("🧬 Hybrid Mode (Meta-Materials)", value=False)
if hybrid_mode:
    st.sidebar.info("Blending Gyroid & Diamond Architectures")
    blend_weight = st.sidebar.slider("Blend Weight (Gyroid <-> Diamond)", 0.0, 1.0, 0.5)
    lattice_type = "Hybrid"
else:
    lattice_type = st.sidebar.selectbox("Lattice Architecture", ["Gyroid", "Diamond"])
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

            # 5. Export for Viewer (Custom Robust Three.js Implementation)
            st.write("Finalizing 3D Scene...")
            VIEWER_PATH = os.path.abspath("temp_viewer.html")
            FALLBACK_PATH = os.path.abspath("fallback_preview.png")
            
            # Save fallback screenshot
            plotter.set_background("#1e1e1e")
            plotter.add_mesh(mesh, color="lightblue", show_edges=True)
            plotter.reset_camera()
            plotter.screenshot(FALLBACK_PATH)
            plotter.close()

            # Generate STL for the viewer
            temp_stl = "temp_view_data.stl"
            save_mesh_to_stl(mesh, temp_stl)
            
            import base64
            with open(temp_stl, "rb") as f:
                stl_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Create a professional Standalone Three.js Viewer
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ImplicitLattice Viewer</title>
                <style>
                    body {{ margin: 0; background-color: #1e1e1e; overflow: hidden; font-family: sans-serif; }}
                    canvas {{ width: 100%; height: 100%; }}
                    #info {{ position: absolute; top: 10px; width: 100%; text-align: center; color: #555; pointer-events: none; }}
                </style>
            </head>
            <body>
                <div id="info">Interactive 3D Preview (Orbit: Left Click, Zoom: Scroll)</div>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
                <script>
                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x1e1e1e);
                    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
                    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    document.body.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;

                    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                    scene.add(ambientLight);
                    const light1 = new THREE.DirectionalLight(0xffffff, 1);
                    light1.position.set(1, 1, 2);
                    scene.add(light1);
                    const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
                    light2.position.set(-1, -1, -2);
                    scene.add(light2);

                    const loader = new THREE.STLLoader();
                    const stlData = atob("{stl_base64}");
                    const bytes = new Uint8Array(stlData.length);
                    for (let i = 0; i < stlData.length; i++) {{
                        bytes[i] = stlData.charCodeAt(i);
                    }}
                    
                    const geometry = loader.parse(bytes.buffer);
                    const material = new THREE.MeshPhongMaterial({{ 
                        color: 0xadd8e6, 
                        specular: 0x111111, 
                        shininess: 100,
                        flatShading: false
                    }});
                    const mesh = new THREE.Mesh(geometry, material);
                    
                    geometry.computeBoundingBox();
                    const center = new THREE.Vector3();
                    geometry.boundingBox.getCenter(center);
                    mesh.position.sub(center);
                    scene.add(mesh);
                    
                    const size = new THREE.Vector3();
                    geometry.boundingBox.getSize(size);
                    const maxDim = Math.max(size.x, size.y, size.z);
                    camera.position.set(maxDim*1.5, maxDim*1.5, maxDim*1.5);
                    camera.lookAt(0, 0, 0);

                    function animate() {{
                        requestAnimationFrame(animate);
                        controls.update();
                        renderer.render(scene, camera);
                    }}
                    animate();

                    window.addEventListener('resize', () => {{
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(window.innerWidth, window.innerHeight);
                    }});
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
            
            status.update(label="✅ Structure Validated & Ready!", state="complete")
        except Exception as e:
            st.error(f"Generation Failed: {str(e)}")
            status.update(label="❌ Error in Synthesis", state="error")

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
        st.info("👈 Configure your lattice and click 'Generate' to visualize the 3D model.")
        
        # If an old fallback exists, we can show it as a teaser or just keep it clean
        if os.path.exists(FALLBACK_PATH):
            st.image(FALLBACK_PATH, caption="Previous Generation Preview", use_container_width=True)

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
