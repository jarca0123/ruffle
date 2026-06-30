// Exact texture passthrough (premultiplied RGBA in, same out). Used to seed an
// offscreen MSAA buffer with a target texture's existing content without the
// un-premultiply / re-premultiply round-trip that `bitmap.frag` performs, which
// would lose 8-bit precision and accumulate across repeated BitmapData.draw.

uniform sampler2D u_texture;

varying vec2 frag_uv;

void main() {
    gl_FragColor = texture2D(u_texture, frag_uv);
}
