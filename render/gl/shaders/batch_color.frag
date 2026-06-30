// Fragment shader for batched solid-color draws: the premultiplied color is
// already final (color transform baked in on the CPU).

varying vec4 frag_color;

void main() {
    gl_FragColor = frag_color;
}
