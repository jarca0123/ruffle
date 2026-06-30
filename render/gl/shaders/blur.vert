// Separable box-blur vertex shader (one pass = one direction). Pre-shifts the
// sampled UV so the center of the first trivially-sampled pixel is at offset 0,
// matching wgpu's fused-sampling blur.

uniform vec4 u_uv_rect; // sampled source region (x, y, w, h), normalized
uniform vec2 u_dir;     // (1/width, 0) for horizontal, (0, 1/height) for vertical
uniform float u_m;      // number of trivially-sampled pixel pairs in the middle

attribute vec2 position;

varying vec2 v_uv;

void main() {
    vec2 uv = u_uv_rect.xy + position * u_uv_rect.zw;
    v_uv = uv - u_dir * u_m;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
