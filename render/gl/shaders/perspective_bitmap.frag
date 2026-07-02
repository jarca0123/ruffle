// Fragment stage of the perspective textured-triangle draw. Divide the linearly
// interpolated (u*t, v*t, t) to get perspective-correct texture coords, then apply
// the same un/re-premultiply color transform as bitmap.frag.

uniform vec4 mult_color;
uniform vec4 add_color;

uniform sampler2D u_texture;

varying vec3 frag_uvt;

void main() {
    vec2 uv = frag_uvt.xy / frag_uvt.z;
    vec4 color = texture2D(u_texture, uv);

    // Unmultiply alpha before applying color transform.
    if (color.a > 0.0) {
        color.rgb /= color.a;
        color = clamp(mult_color * color + add_color, 0.0, 1.0);
        float alpha = clamp(color.a, 0.0, 1.0);
        color = vec4(color.rgb * alpha, alpha);
    }

    gl_FragColor = color;
}
