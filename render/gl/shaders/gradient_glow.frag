// Flash GradientGlowFilter: like GlowFilter, but the glow color at each pixel is
// looked up from a 256-entry gradient ramp (premultiplied) indexed by the
// blurred-alpha intensity, instead of a single color. Reuses glow.vert.

uniform sampler2D u_texture; // source
uniform sampler2D u_blurred; // blurred source (region-sized)
uniform sampler2D u_ramp;    // 256x1 premultiplied gradient ramp
uniform float u_strength;
uniform int u_type;          // 0 outer, 1 inner, 2 full (on top)
uniform int u_knockout;
uniform int u_composite_source;

varying vec2 v_source_uv;
varying vec2 v_blur_uv;

void main() {
    float blur = texture2D(u_blurred, v_blur_uv).a;
    vec4 dest = texture2D(u_texture, v_source_uv);
    if (v_blur_uv.x < 0.0 || v_blur_uv.x > 1.0 || v_blur_uv.y < 0.0 || v_blur_uv.y > 1.0) {
        blur = 0.0;
    }

    vec4 result;
    if (u_type == 1) {
        // inner: glow inside the object
        float t = clamp((1.0 - blur) * u_strength, 0.0, 1.0);
        vec4 glow = texture2D(u_ramp, vec2(t, 0.5));
        if (u_knockout != 0) {
            result = glow * dest.a;
        } else if (u_composite_source != 0) {
            result = glow * dest.a + dest * (1.0 - glow.a);
        } else {
            result = glow * dest.a;
        }
    } else if (u_type == 2) {
        // full: glow on top of the object
        float t = clamp(blur * u_strength, 0.0, 1.0);
        vec4 glow = texture2D(u_ramp, vec2(t, 0.5));
        if (u_knockout != 0) {
            result = glow;
        } else {
            result = glow + dest * (1.0 - glow.a);
        }
    } else {
        // outer: glow outside the object
        float t = clamp(blur * u_strength, 0.0, 1.0);
        vec4 glow = texture2D(u_ramp, vec2(t, 0.5));
        if (u_knockout != 0) {
            result = glow * (1.0 - dest.a);
        } else if (u_composite_source != 0) {
            result = glow * (1.0 - dest.a) + dest;
        } else {
            result = glow;
        }
    }

    gl_FragColor = result;
}
