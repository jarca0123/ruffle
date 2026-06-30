// Fullscreen-quad vertex shader for filters. `position` is in [0,1]; the quad
// covers the whole output, and `v_uv` samples the source region given by
// `u_uv_rect` (x, y, width, height in normalized source-texture coordinates).
// No vertical flip: output texel row 0 maps to the top of the source region,
// keeping offscreen textures consistent (Flash-top at texel row 0).

uniform vec4 u_uv_rect;

attribute vec2 position;

varying vec2 v_uv;

void main() {
    v_uv = u_uv_rect.xy + position * u_uv_rect.zw;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
