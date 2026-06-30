// Complex Flash blend modes (port of wgpu's blend/*.wgsl). Samples the parent
// (destination, already composited into the framebuffer) and the current
// (source) region textures, both premultiplied, and outputs the premultiplied
// blended pixel. Where the source is empty we discard, leaving the framebuffer's
// existing parent pixel untouched.

uniform sampler2D u_current; // src
uniform sampler2D u_parent;  // dst
// 0 multiply, 1 lighten, 2 darken, 3 difference, 4 invert,
// 5 alpha, 6 erase, 7 overlay, 8 hardlight
uniform int u_blend_mode;

varying vec2 v_uv;

// Per-channel overlay/hardlight: ctrl<=0.5 ? 2*a*b : 1 - 2*(1-a)*(1-b).
// `step(ctrl, 0.5)` is 1.0 where ctrl<=0.5 (GLSL ES 1.00 compatible; avoids the
// bvec form of mix()).
vec3 overlay_hardlight(vec3 ctrl, vec3 a, vec3 b) {
    vec3 lo = 2.0 * a * b;
    vec3 hi = 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
    vec3 sel = step(ctrl, vec3(0.5));
    return mix(hi, lo, sel);
}

void main() {
    vec4 dst = texture2D(u_parent, v_uv);
    vec4 src = texture2D(u_current, v_uv);

    // Matches Flash / wgpu: a fully-transparent source writes nothing.
    if (src.a <= 0.0) {
        discard;
    }

    vec4 outc;
    if (u_blend_mode == 5) {
        // alpha: keep the parent color, masked by the source alpha.
        outc = vec4(dst.rgb * src.a, src.a * dst.a);
    } else if (u_blend_mode == 6) {
        // erase: cut the source alpha out of the parent.
        outc = vec4(dst.rgb * (1.0 - src.a), (1.0 - src.a) * dst.a);
    } else {
        // Un-premultiply for the blend function (dst guarded for dst.a == 0).
        vec3 s = src.rgb / src.a;
        vec3 d = dst.a > 0.0 ? dst.rgb / dst.a : vec3(0.0);
        vec3 bf;
        if (u_blend_mode == 0) {
            bf = s * d;
        } else if (u_blend_mode == 1) {
            bf = max(s, d);
        } else if (u_blend_mode == 2) {
            bf = min(s, d);
        } else if (u_blend_mode == 3) {
            bf = abs(d - s);
        } else if (u_blend_mode == 4) {
            bf = 1.0 - d;
        } else if (u_blend_mode == 7) {
            bf = overlay_hardlight(d, s, d); // overlay: control by dst
        } else {
            bf = overlay_hardlight(s, s, d); // hardlight: control by src
        }
        outc = vec4(
            src.rgb * (1.0 - dst.a) + dst.rgb * (1.0 - src.a) + src.a * dst.a * bf,
            src.a + dst.a * (1.0 - src.a)
        );
    }

    gl_FragColor = outc;
}
