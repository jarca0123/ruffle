// Batched bitmap draw: positions are already world-transformed on the CPU and
// the texture coordinates precomputed, so only the view projection remains. The
// fragment stage reuses bitmap.frag (color transform via per-batch uniforms).

uniform mat4 view_matrix;

attribute vec2 position;
attribute vec2 uv;

varying vec2 frag_uv;

void main() {
    frag_uv = uv;
    gl_Position = view_matrix * vec4(position, 0.0, 1.0);
}
