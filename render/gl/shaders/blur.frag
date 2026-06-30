// Separable box blur with fused bilinear sampling (port of wgpu's blur.wgsl).
// Operates on premultiplied RGBA (a box average is linear, so no un-premultiply).
//
// The middle loop uses a constant iteration cap with an early `break`, which is
// valid in GLSL ES 1.00 (WebGL1) — where dynamic loop bounds are forbidden — as
// well as on WebGL2/desktop GL. The cap covers Flash's maximum blur (255 px:
// radius <= 127, so m2 <= ~252, i.e. <= 126 iterations).

uniform sampler2D u_texture;
uniform vec2 u_dir;
uniform float u_full_size;
uniform float u_m2;
uniform float u_first_weight;
uniform float u_last_offset;
uniform float u_last_weight;

varying vec2 v_uv;

const int MAX_ITERS = 130;

void main() {
    vec4 total = texture2D(u_texture, v_uv - u_dir) * u_first_weight;

    vec4 center = vec4(0.0);
    for (int k = 0; k < MAX_ITERS; k++) {
        float i = float(k) * 2.0 + 0.5;
        if (i >= u_m2) {
            break;
        }
        // Sample between two texels and rely on bilinear filtering to average
        // them (weight 1+1).
        center += texture2D(u_texture, v_uv + u_dir * i);
    }
    total += center * 2.0;

    vec2 last_location = v_uv + u_dir * (u_m2 + u_last_offset);
    total += texture2D(u_texture, last_location) * u_last_weight;

    vec4 result = total / u_full_size;

    // Round to imitate Flash Player's fixed-point math.
    gl_FragColor = floor(result * 255.0) / 255.0;
}
