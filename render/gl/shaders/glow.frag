// Glow / DropShadow composite (port of wgpu's glow.wgsl). Combines the source
// with its blurred alpha, colorized by u_color and scaled by u_strength.
// Output is premultiplied RGBA. u_color is straight (un-premultiplied) RGBA.

uniform sampler2D u_texture; // source
uniform sampler2D u_blurred; // blurred source (region-sized)
uniform vec4 u_color;
uniform float u_strength;
uniform int u_inner;
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

    vec4 color = vec4(u_color.r, u_color.g, u_color.b, 1.0);
    if (u_inner != 0) {
        float alpha = u_color.a * clamp((1.0 - blur) * u_strength, 0.0, 1.0);
        if (u_knockout != 0) {
            color = color * alpha * dest.a;
        } else if (u_composite_source != 0) {
            color = color * alpha * dest.a + dest * (1.0 - alpha);
        } else {
            color = color * alpha * dest.a;
        }
    } else {
        float alpha = u_color.a * clamp(blur * u_strength, 0.0, 1.0);
        if (u_knockout != 0) {
            color = color * alpha * (1.0 - dest.a);
        } else if (u_composite_source != 0) {
            color = color * alpha * (1.0 - dest.a) + dest;
        } else {
            color = color * alpha;
        }
    }

    gl_FragColor = color;
}
