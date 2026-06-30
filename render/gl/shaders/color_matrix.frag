// ColorMatrix filter: un-premultiply, apply the 4x5 matrix, clamp, re-premultiply.
// u_cm layout matches swf::ColorMatrixFilter.matrix ([f32; 20], row-major):
//   row r: [r_to_r, g_to_r, b_to_r, a_to_r, r_extra], then g, b, a rows.

uniform sampler2D u_texture;
uniform float u_cm[20];

varying vec2 v_uv;

void main() {
    vec4 src = texture2D(u_texture, v_uv);
    float a = src.a;
    vec3 c = a > 0.0 ? src.rgb / a : vec3(0.0);

    float r = clamp(u_cm[0] * c.r + u_cm[1] * c.g + u_cm[2] * c.b + u_cm[3] * a + u_cm[4] / 255.0, 0.0, 1.0);
    float g = clamp(u_cm[5] * c.r + u_cm[6] * c.g + u_cm[7] * c.b + u_cm[8] * a + u_cm[9] / 255.0, 0.0, 1.0);
    float b = clamp(u_cm[10] * c.r + u_cm[11] * c.g + u_cm[12] * c.b + u_cm[13] * a + u_cm[14] / 255.0, 0.0, 1.0);
    float oa = clamp(u_cm[15] * c.r + u_cm[16] * c.g + u_cm[17] * c.b + u_cm[18] * a + u_cm[19] / 255.0, 0.0, 1.0);

    gl_FragColor = vec4(r * oa, g * oa, b * oa, oa);
}
