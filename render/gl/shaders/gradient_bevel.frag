// Flash GradientBevelFilter: like BevelFilter, but the rim color comes from a
// 256-entry gradient ramp (premultiplied). The signed bevel intensity is mapped
// to [0,1] (0.5 = no bevel, <0.5 shadow side, >0.5 highlight side) and used as
// the ramp index. Reuses bevel.vert.

uniform sampler2D u_texture; // source
uniform sampler2D u_blurred; // blurred source alpha (region-sized)
uniform sampler2D u_ramp;    // 256x1 premultiplied gradient ramp
uniform float u_strength;
uniform int u_bevel_type;    // 0 outer, 1 inner, 2 full
uniform int u_knockout;

varying vec2 v_source_uv;
varying vec2 v_blur_left;
varying vec2 v_blur_right;

void main() {
    float blur_left = texture2D(u_blurred, v_blur_left).a;
    float blur_right = texture2D(u_blurred, v_blur_right).a;
    vec4 dest = texture2D(u_texture, v_source_uv);

    if (v_blur_left.x < 0.0 || v_blur_left.x > 1.0 || v_blur_left.y < 0.0 || v_blur_left.y > 1.0) {
        blur_left = 0.0;
    }
    if (v_blur_right.x < 0.0 || v_blur_right.x > 1.0 || v_blur_right.y < 0.0 || v_blur_right.y > 1.0) {
        blur_right = 0.0;
    }

    bool outer = u_bevel_type == 0 || u_bevel_type == 2;
    bool inner = u_bevel_type == 1 || u_bevel_type == 2;

    float h = clamp((blur_left - blur_right) * u_strength, 0.0, 1.0);
    float s = clamp((blur_right - blur_left) * u_strength, 0.0, 1.0);
    float t = clamp(0.5 + (h - s) * 0.5, 0.0, 1.0);
    vec4 glow = texture2D(u_ramp, vec2(t, 0.5));

    vec4 color;
    if (inner && outer) {
        color = (u_knockout != 0) ? glow : dest - dest * glow.a + glow;
    } else if (inner) {
        color = (u_knockout != 0) ? glow * dest.a : glow * dest.a + dest * (1.0 - glow.a);
    } else {
        color = (u_knockout != 0) ? glow - glow * dest.a : dest + glow - glow * dest.a;
    }

    gl_FragColor = color;
}
