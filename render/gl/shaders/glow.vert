// Vertex shader for the Glow / DropShadow composite pass. Emits the source UV
// (the region within the source texture) and the blur UV (the region-sized
// blurred texture, shifted by the shadow offset).

uniform vec4 u_uv_rect;     // source region (x, y, w, h), normalized
uniform vec2 u_blur_offset; // shadow offset in normalized output coords

attribute vec2 position;

varying vec2 v_source_uv;
varying vec2 v_blur_uv;

void main() {
    v_source_uv = u_uv_rect.xy + position * u_uv_rect.zw;
    v_blur_uv = position + u_blur_offset;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
