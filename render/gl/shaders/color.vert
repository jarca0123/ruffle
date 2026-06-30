// NOTE: No `#version` / `precision` here — a target-specific preamble is
// prepended at compile time (see `shader.rs`). Bodies are written in GLSL ES
// 1.00 style (`attribute`/`varying`/`gl_FragColor`/`texture2D`).

uniform mat4 view_matrix;
uniform mat4 world_matrix;
uniform vec4 mult_color;
uniform vec4 add_color;

attribute vec2 position;
attribute vec4 color;
varying vec4 frag_color;

void main() {
    frag_color = clamp(color * mult_color + add_color, 0.0, 1.0);
    float alpha = clamp(frag_color.a, 0.0, 1.0);
    frag_color = vec4(frag_color.rgb * alpha, alpha);
    gl_Position = view_matrix * world_matrix * vec4(position, 0.0, 1.0);
}
