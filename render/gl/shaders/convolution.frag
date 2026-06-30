// Flash ConvolutionFilter: an N x M kernel applied over the source region.
// `result = sum(kernel[i] * sample_i) / divisor + bias`, optionally preserving
// the source alpha. Out-of-region taps use the clamped edge (CLAMP) or the
// default color. Operates on premultiplied RGBA (an approximation of Flash's
// straight-alpha convolution, exact for opaque content).
//
// Kept GLSL ES 1.00 friendly: a constant-bound loop with `break`, the kernel
// indexed only by the loop variable, and a single unconditional texture sample.

uniform sampler2D u_texture;
uniform vec4 u_uv_rect;        // source region (x, y, w, h), normalized
uniform vec2 u_texel;          // (1/src_w, 1/src_h)
uniform float u_kernel[49];    // row-major, padded to MAX_TAPS
uniform float u_cols;
uniform float u_rows;
uniform float u_divisor;
uniform float u_bias;          // normalized (Flash bias / 255)
uniform vec4 u_default_color;  // premultiplied
uniform float u_clamp;         // 1.0 = clamp edges, 0.0 = use default color
uniform float u_preserve_alpha;

varying vec2 v_uv;

const int MAX_TAPS = 49;

void main() {
    vec2 region_min = u_uv_rect.xy;
    vec2 region_max = u_uv_rect.xy + u_uv_rect.zw;
    float cx = (u_cols - 1.0) * 0.5;
    float cy = (u_rows - 1.0) * 0.5;
    float count = u_cols * u_rows;

    vec4 acc = vec4(0.0);
    for (int i = 0; i < MAX_TAPS; i++) {
        if (float(i) >= count) {
            break;
        }
        float fi = float(i);
        float r = floor(fi / u_cols);
        float c = fi - r * u_cols;
        vec2 nuv = v_uv + (vec2(c, r) - vec2(cx, cy)) * u_texel;

        bool oob = nuv.x < region_min.x || nuv.x > region_max.x
            || nuv.y < region_min.y || nuv.y > region_max.y;
        // Sample at a clamped coordinate (safe, unconditional); override with the
        // default color for out-of-region taps when not clamping.
        vec4 s = texture2D(u_texture, clamp(nuv, region_min, region_max));
        if (oob && u_clamp < 0.5) {
            s = u_default_color;
        }
        acc += u_kernel[i] * s;
    }

    vec4 result = acc / u_divisor + vec4(u_bias);
    if (u_preserve_alpha > 0.5) {
        result.a = texture2D(u_texture, v_uv).a;
    }
    gl_FragColor = clamp(result, 0.0, 1.0);
}
