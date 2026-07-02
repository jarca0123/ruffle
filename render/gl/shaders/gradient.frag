uniform mat4 view_matrix;
uniform mat4 world_matrix;
uniform vec4 mult_color;
uniform vec4 add_color;
uniform mat3 u_matrix;

uniform int u_gradient_type;
uniform int u_repeat_mode;
uniform float u_focal_point;
uniform int u_interpolation;
// Baked 256x1 gradient ramp (see `Gradient::ramp`). Sampled with hardware
// linear filtering, matching wgpu's gradient texture.
uniform sampler2D u_texture;

varying vec2 frag_uv;

// Matches wgpu's `common__linear_to_srgb`: the ramp stores straight (un-
// premultiplied) colors, so un-premultiply by alpha before the transfer curve
// and re-premultiply after, exactly as wgpu does. This is what makes
// semi-transparent gradient stops match bit-for-bit.
vec4 linear_to_srgb(vec4 c) {
    vec3 rgb = c.rgb;
    if (c.a > 0.0) {
        rgb = rgb / c.a;
    }
    vec3 a = 12.92 * rgb;
    vec3 b = 1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055;
    vec3 sel = step(vec3(0.0031308), rgb);
    return vec4(mix(a, b, sel) * c.a, c.a);
}

void main() {
    float t;
    if (u_gradient_type == 0) {
        t = frag_uv.x;
    } else if (u_gradient_type == 1) {
        t = length(frag_uv * 2.0 - 1.0);
    } else if (u_gradient_type == 2) {
        vec2 uv = frag_uv * 2.0 - 1.0;
        vec2 d = vec2(u_focal_point, 0.0) - uv;
        float l = length(d);
        d /= l;
        t = l / (sqrt(1.0 -  u_focal_point*u_focal_point*d.y*d.y) + u_focal_point*d.x);
    }
    if (u_repeat_mode == 0) {
        // Clamp
        t = clamp(t, 0.0, 1.0);
    } else if (u_repeat_mode == 1) {
        // Repeat
        t = fract(t);
    } else {
        // Mirror
        if (t < 0.0) {
            t = -t;
        }

        if (int(mod(t, 2.0)) == 0) {
            t = fract(t);
        } else {
            t = 1.0 - fract(t);
        }
    }

    // Sample the baked ramp. `t` is already folded into [0, 1] above, so the
    // texture wrap mode is irrelevant (clamp). The 256-texel quantization plus
    // hardware linear filtering matches wgpu exactly.
    vec4 color = texture2D(u_texture, vec2(t, 0.0));

    if (u_interpolation != 0) {
        color = linear_to_srgb(color);
    }

    // Color transform is applied after sampling/interpolation, matching wgpu.
    color = clamp(mult_color * color + add_color, 0.0, 1.0);

    float alpha = clamp(color.a, 0.0, 1.0);
    gl_FragColor = vec4(color.rgb * alpha, alpha);
}
