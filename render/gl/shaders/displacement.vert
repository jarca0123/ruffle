// Vertex shader for the DisplacementMap filter. Emits region-local UVs in
// [0, 1]; the fragment shader works in region pixel space and maps the final
// sample back into the source texture region via u_uv_rect.

attribute vec2 position;

varying vec2 v_region_uv;

void main() {
    v_region_uv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
