// Perspective-correct textured triangle (Graphics.drawTriangles with a
// 3-component uvtData). Positions are shape-space pixels (world-transformed by the
// usual view/world matrices); `uvt` is (u, v, t = 1/w). We pass (u*t, v*t, t) — in
// 2D gl_Position.w is 1, so varyings interpolate LINEARLY in screen space, and the
// fragment shader divides to recover perspective-correct (u, v).

uniform mat4 view_matrix;
uniform mat4 world_matrix;

attribute vec2 position;
attribute vec3 uvt;

varying vec3 frag_uvt;

void main() {
    frag_uvt = vec3(uvt.x * uvt.z, uvt.y * uvt.z, uvt.z);
    gl_Position = view_matrix * world_matrix * vec4(position, 0.0, 1.0);
}
