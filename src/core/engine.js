// WebGL Engine for ImplicitLattice Rendering
// Sets up scene, lighting, controls, uniform mappings, and handles canvas resize.

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Lighting Setup
const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
scene.add(ambientLight);
const light1 = new THREE.DirectionalLight(0xffffff, 1.0);
light1.position.set(1, 1, 2);
scene.add(light1);

// Decode base64 STL data passed from Streamlit backend
const loader = new THREE.STLLoader();
const stlData = atob("__STL_BASE64__");
const bytes = new Uint8Array(stlData.length);
for (let i = 0; i < stlData.length; i++) {
    bytes[i] = stlData.charCodeAt(i);
}

const geometry = loader.parse(bytes.buffer);

// Shader Material setup with Explicit Uniform Mappings
const material = new THREE.ShaderMaterial({
    uniforms: {
        tMin: { value: __T_MIN__ },
        tMax: { value: __T_MAX__ },
        frequency: { value: __FREQUENCY__ },
        mode: { value: __GRADING_MODE__ },
        attPos: { value: new THREE.Vector3(__ATT_X__, __ATT_Y__, __ATT_Z__) },
        attRad: { value: __ATT_RAD__ },
        zMin: { value: __Z_MIN__ },
        zMax: { value: __Z_MAX__ },
        latticeType: { value: __LATTICE_TYPE__ },
        blendWeight: { value: __BLEND_WEIGHT__ }
    },
    // Shaders read from external .glsl files in Python and loaded here
    vertexShader: `__VERTEX_SHADER__`,
    fragmentShader: `__FRAGMENT_SHADER__`
});

const mesh = new THREE.Mesh(geometry, material);

// Center the mesh in the viewport
geometry.computeBoundingBox();
const center = new THREE.Vector3();
geometry.boundingBox.getCenter(center);
mesh.position.sub(center);
scene.add(mesh);

// Adjust camera coordinates according to geometry size
const size = new THREE.Vector3();
geometry.boundingBox.getSize(size);
const maxDim = Math.max(size.x, size.y, size.z);
camera.position.set(maxDim * 1.5, maxDim * 1.5, maxDim * 1.5);
camera.lookAt(0, 0, 0);

// Dynamic Canvas Resize Handler
window.addEventListener('resize', onWindowResize, false);

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Rendering Animation Loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
