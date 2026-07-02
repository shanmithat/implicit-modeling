varying vec3 vPos;

// --- Uniform Mapping & Mathematical Parameter Exposure ---
// tMin: Minimum thickness limit (maps to the lower bounds of functional grading)
uniform float tMin;
// tMax: Maximum thickness limit (maps to the upper bounds of functional grading)
uniform float tMax;

// mode: Grading Strategy Selector
// 0 = Linear Z-Grading: Thickness varies linearly across the vertical range
//     Math: f(z) = (z - zMin) / (zMax - zMin)
// 1 = Point Attractor: Thickness reinforced locally around a coordinate
//     Math: f(x,y,z) = 1.0 - clamp(distance(P, attPos) / attRad, 0.0, 1.0)
uniform int mode;

// attPos: Attractor coordinates (x_a, y_a, z_a) for localized reinforcement
uniform vec3 attPos;
// attRad: Radial influence limit (influence radius R) of the attractor point
uniform float attRad;

// zMin: Spatial lower bound of the container mesh along the Z-axis
uniform float zMin;
// zMax: Spatial upper bound of the container mesh along the Z-axis
uniform float zMax;

/**
 * Mathematical formulation of the TPMS (Triply Periodic Minimal Surfaces) supported:
 * 
 * 1. Gyroid Level Set Equation:
 *    F_gyroid(x, y, z) = sin(w*x)*cos(w*y) + sin(w*y)*cos(w*z) + sin(w*z)*cos(w*x) = t
 *    Where w = 2*pi / CellSize defines the spatial frequency, and t is the thickness offset.
 * 
 * 2. Diamond (Schwarz D) Level Set Equation:
 *    F_diamond(x, y, z) = sin(w*x)*sin(w*y)*sin(w*z) + sin(w*x)*cos(w*y)*cos(w*z) 
 *                         + cos(w*x)*sin(w*y)*cos(w*z) + cos(w*x)*cos(w*y)*sin(w*z) = t
 * 
 * 3. Functional Grading (Multi-Scale Interface):
 *    The thickness iso-level t varies continuously in space: t = t(x, y, z).
 *    This shader maps the local spatial position vPos to a normalized grading parameter
 *    which drives the visual colormap gradient corresponding to the physical thickness.
 */

// Plasma colormap for thickness visualization
vec3 plasma(float t) {
    const vec3 c1 = vec3(0.05, 0.03, 0.53);
    const vec3 c2 = vec3(0.61, 0.09, 0.62);
    const vec3 c3 = vec3(0.93, 0.47, 0.33);
    const vec3 c4 = vec3(0.94, 0.98, 0.13);
    if (t < 0.33) return mix(c1, c2, t * 3.0);
    if (t < 0.66) return mix(c2, c3, (t - 0.33) * 3.0);
    return mix(c3, c4, (t - 0.66) * 3.0);
}

void main() {
    float val;
    
    if (mode == 1) {
        // Point Attractor Grading Mode
        // Compute Euclidean distance from the fragment position vPos to the attractor coordinate attPos
        // Math: d = ||vPos - attPos||
        float d = distance(vPos, attPos);
        
        // Normalize distance by influence radius and invert so it is thicker (1.0) near the attractor center
        // Math: val = 1.0 - clamp(d / attRad, 0.0, 1.0)
        val = clamp(d / attRad, 0.0, 1.0);
        val = 1.0 - val;
    } else {
        // Linear Z-Grading Mode
        // Map the fragment's Z coordinate relative to the container's vertical spatial bounds
        // Math: val = (vPos.z - zMin) / (zMax - zMin)
        val = clamp((vPos.z - zMin) / (zMax - zMin), 0.0, 1.0);
    }
    
    // Output the mapped plasma color corresponding to the localized lattice thickness grading
    gl_FragColor = vec4(plasma(val), 1.0);
}
