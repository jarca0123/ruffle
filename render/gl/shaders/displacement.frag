// Flash DisplacementMapFilter (port of wgpu's displacement_map.wgsl). A map
// texture's selected channels offset the source sampling coordinate. Works in
// region pixel space (source size = region size), then maps the displaced
// coordinate back into the source texture region via u_uv_rect. Output is
// premultiplied RGBA.

uniform sampler2D u_source;
uniform sampler2D u_map;
uniform vec4 u_uv_rect;       // source region (x, y, w, h), normalized
uniform vec4 u_color;         // straight RGBA (color mode)
uniform vec2 u_components;    // (x_channel, y_channel): 1=r 2=g 4=b 8=a
uniform float u_mode;         // 0 wrap, 1 clamp, 2 ignore, 3 color
uniform vec2 u_scale;         // scale_x, scale_y
uniform vec2 u_source_size;   // region pixels
uniform vec2 u_map_size;      // map texture pixels
uniform vec2 u_offset;        // map_point
uniform vec2 u_viewscale;     // viewscale_x, viewscale_y

varying vec2 v_region_uv;

float get_component(vec4 m, float comp) {
    if (comp == 1.0) return m.r * 255.0;
    if (comp == 2.0) return m.g * 255.0;
    if (comp == 4.0) return m.b * 255.0;
    if (comp == 8.0) return m.a * 255.0;
    return 128.0; // zero displacement
}

void main() {
    vec2 source_pos = v_region_uv * u_source_size;
    vec2 map_uv = vec2(
        (source_pos.x - u_offset.x) / u_viewscale.x / u_map_size.x,
        (source_pos.y - u_offset.y) / u_viewscale.y / u_map_size.y
    );
    vec4 m = texture2D(u_map, map_uv);
    if (map_uv.x < 0.0 || map_uv.x > 1.0 || map_uv.y < 0.0 || map_uv.y > 1.0) {
        m = vec4(0.5);
    }

    vec2 eff_scale = u_viewscale * u_scale;
    vec2 displaced = vec2(
        source_pos.x + (get_component(m, u_components.x) - 128.0) * eff_scale.x / 256.0,
        source_pos.y + (get_component(m, u_components.y) - 128.0) * eff_scale.y / 256.0
    );
    vec2 duv = displaced / u_source_size;
    bool oob = duv.x < 0.0 || duv.x > 1.0 || duv.y < 0.0 || duv.y > 1.0;

    if (u_mode == 0.0) {                  // wrap
        duv = fract(duv);
    } else if (u_mode == 1.0) {           // clamp
        duv = clamp(duv, 0.0, 1.0);
    } else if (u_mode == 2.0 && oob) {    // ignore
        duv = v_region_uv;
    }

    vec2 src_uv = u_uv_rect.xy + duv * u_uv_rect.zw;
    vec4 result = texture2D(u_source, src_uv);
    if (u_mode == 3.0 && oob) {           // color
        result = vec4(u_color.rgb, 1.0) * u_color.a;
    }
    gl_FragColor = result;
}
