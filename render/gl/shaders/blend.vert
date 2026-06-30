// Fullscreen-quad vertex shader for the complex-blend composite. The quad
// covers the whole viewport (set to the blended region in framebuffer pixels);
// v_uv samples the region-sized src/parent textures directly.

attribute vec2 position;

varying vec2 v_uv;

void main() {
    v_uv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
