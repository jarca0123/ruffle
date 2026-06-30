// Batched solid-color draw: positions are already world-transformed and colors
// already have the color transform baked in and are premultiplied (done on the
// CPU when appending to the batch). Only the view projection remains.

uniform mat4 view_matrix;

attribute vec2 position;
attribute vec4 color;

varying vec4 frag_color;

void main() {
    frag_color = color;
    gl_Position = view_matrix * vec4(position, 0.0, 1.0);
}
