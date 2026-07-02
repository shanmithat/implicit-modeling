varying vec3 vPos;

void main() {
    // Transform position to world coordinates to pass to the fragment shader
    vPos = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
