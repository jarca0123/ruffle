// Vertex shader for the Bevel composite pass (port of wgpu's bevel.wgsl). Emits
// the source UV (the region within the source texture) and two blur UVs: the
// region-sized blurred alpha shifted by +offset (highlight side) and -offset
// (shadow side).

uniform vec4 u_uv_rect;     // source region (x, y, w, h), normalized
uniform vec2 u_blur_offset; // bevel offset in normalized output coords

attribute vec2 position;

varying vec2 v_source_uv;
varying vec2 v_blur_left;
varying vec2 v_blur_right;

void main() {
    v_source_uv = u_uv_rect.xy + position * u_uv_rect.zw;
    v_blur_left = position + u_blur_offset;
    v_blur_right = position - u_blur_offset;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
