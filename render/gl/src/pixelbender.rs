//! Translates a PixelBender shader (parsed bytecode) into a GLSL ES 1.00
//! fragment shader plus the metadata needed to bind parameters/inputs at run
//! time. Ported from the naga IR translator (`render/naga-pixelbender`), but
//! emitting GLSL directly so it runs on the oldest GL (WebGL1/GLES2) — naga's
//! GLSL backend only targets ES 3.00+.
//!
//! Model (matching the naga port): every register is a 4-component vector
//! (`vec4` float bank / `ivec4` int bank). Reads swizzle-and-pad to a `vec4`;
//! writes mask only the destination channels, consuming source components
//! positionally.

use ruffle_render::pixel_bender::{
    Opcode, Operation, PixelBenderParam, PixelBenderParamQualifier, PixelBenderReg,
    PixelBenderRegChannel, PixelBenderRegKind, PixelBenderShader, PixelBenderTypeOpcode,
    OUT_COORD_NAME,
};
use std::collections::BTreeSet;
use std::fmt::Write;

/// Where a value parameter lives in the uniform arrays (assigned in param
/// declaration order, matching the running-counter layout the run path uses).
#[derive(Clone, Copy, Debug)]
pub struct ParamSlot {
    pub offset: usize,
    pub is_int: bool,
    /// Number of vec4 slots (1 for scalars/vectors/mat2, 3 for mat3, 4 for mat4).
    pub num_vec4s: usize,
}

pub struct TranslatedShader {
    pub glsl: String,
    /// Float uniform array length (>= 1).
    pub float_slots: usize,
    /// Int uniform array length (>= 1).
    pub int_slots: usize,
    /// Texture-input indices used (their `param.index`).
    pub inputs: Vec<u8>,
    /// Input indices that are sampled with nearest filtering.
    pub nearest_inputs: Vec<u8>,
    /// Per-`shader.params` slot, `None` for texture/output/outCoord params.
    pub param_slots: Vec<Option<ParamSlot>>,
    pub output_channels: usize,
}

fn chan_letter(c: PixelBenderRegChannel) -> char {
    match c {
        PixelBenderRegChannel::R => 'x',
        PixelBenderRegChannel::G => 'y',
        PixelBenderRegChannel::B => 'z',
        PixelBenderRegChannel::A => 'w',
        // Matrix channels are handled separately; shouldn't reach here.
        _ => 'x',
    }
}

fn reg_name(reg: &PixelBenderReg) -> String {
    match reg.kind {
        PixelBenderRegKind::Float => format!("ff{}", reg.index),
        PixelBenderRegKind::Int => format!("ii{}", reg.index),
    }
}

/// Swizzle letters for the register's channels, padded to 4 by repeating the
/// last (padding lanes are discarded by the destination write-mask).
fn swizzle_padded(reg: &PixelBenderReg) -> String {
    let mut s: String = reg.channels.iter().map(|&c| chan_letter(c)).collect();
    if s.is_empty() {
        s.push('x');
    }
    let last = s.chars().last().unwrap();
    while s.len() < 4 {
        s.push(last);
    }
    s
}

/// Swizzle letters for exactly the register's channel count (2/3/4 vec).
fn swizzle_exact(reg: &PixelBenderReg) -> String {
    reg.channels.iter().map(|&c| chan_letter(c)).collect()
}

/// `reg.<padded swizzle>` — a `vec4`/`ivec4` rvalue.
fn load_padded(reg: &PixelBenderReg) -> String {
    format!("{}.{}", reg_name(reg), swizzle_padded(reg))
}

/// `reg.<exact swizzle>` — a `vecN` rvalue (N = channel count).
fn load_exact(reg: &PixelBenderReg) -> String {
    let n = reg.channels.len().max(1);
    if n == 1 {
        format!("{}.{}", reg_name(reg), chan_letter(reg.channels[0]))
    } else {
        format!("{}.{}", reg_name(reg), swizzle_exact(reg))
    }
}

const XYZW: [char; 4] = ['x', 'y', 'z', 'w'];

/// Emits `dst.<mask> = <cast>(value).<first-k>;` — stores the leading `k`
/// components of `value` into the destination's masked lanes.
fn store(out: &mut String, dst: &PixelBenderReg, value: &str) {
    let mask: String = dst.channels.iter().map(|&c| chan_letter(c)).collect();
    let k = dst.channels.len().max(1);
    let src_sw: String = XYZW[..k].iter().collect();
    // Cast the value to the destination bank's type.
    let cast = match dst.kind {
        PixelBenderRegKind::Float => "vec4",
        PixelBenderRegKind::Int => "ivec4",
    };
    let mask = if mask.is_empty() { "x".to_string() } else { mask };
    let _ = writeln!(
        out,
        "    {}.{} = {}({}).{};",
        reg_name(dst),
        mask,
        cast,
        value,
        src_sw
    );
}

/// Number of vec4 slots a value param consumes.
fn param_slot_count(ty: PixelBenderTypeOpcode) -> usize {
    match ty {
        PixelBenderTypeOpcode::TFloat3x3 => 3,
        PixelBenderTypeOpcode::TFloat4x4 => 4,
        _ => 1,
    }
}

fn param_is_int(ty: PixelBenderTypeOpcode) -> bool {
    matches!(
        ty,
        PixelBenderTypeOpcode::TInt
            | PixelBenderTypeOpcode::TInt2
            | PixelBenderTypeOpcode::TInt3
            | PixelBenderTypeOpcode::TInt4
            | PixelBenderTypeOpcode::TBool
            | PixelBenderTypeOpcode::TBool2
            | PixelBenderTypeOpcode::TBool3
            | PixelBenderTypeOpcode::TBool4
    )
}

pub fn translate(shader: &PixelBenderShader) -> Result<TranslatedShader, String> {
    let mut used_float: BTreeSet<u32> = BTreeSet::new();
    let mut used_int: BTreeSet<u32> = BTreeSet::new();
    let mut inputs: Vec<u8> = Vec::new();

    let mark = |reg: &PixelBenderReg,
                uf: &mut BTreeSet<u32>,
                ui: &mut BTreeSet<u32>| {
        // Matrix registers span consecutive registers.
        let span = match reg.channels.first() {
            Some(PixelBenderRegChannel::M3x3) => 3,
            Some(PixelBenderRegChannel::M4x4) => 4,
            _ => 1,
        };
        for i in 0..span {
            match reg.kind {
                PixelBenderRegKind::Float => {
                    uf.insert(reg.index + i);
                }
                PixelBenderRegKind::Int => {
                    ui.insert(reg.index + i);
                }
            }
        }
    };

    // Assign parameter slots + collect used registers from params.
    let mut param_slots: Vec<Option<ParamSlot>> = vec![None; shader.params.len()];
    let mut float_offset = 0usize;
    let mut int_offset = 0usize;
    let mut out_coord: Option<PixelBenderReg> = None;
    for (i, param) in shader.params.iter().enumerate() {
        match param {
            PixelBenderParam::Normal {
                qualifier,
                param_type,
                reg,
                name,
                ..
            } => {
                mark(reg, &mut used_float, &mut used_int);
                if name == OUT_COORD_NAME {
                    out_coord = Some(reg.clone());
                    continue;
                }
                if matches!(qualifier, PixelBenderParamQualifier::Output) {
                    continue;
                }
                if matches!(param_type, PixelBenderTypeOpcode::TString) {
                    continue;
                }
                let is_int = param_is_int(*param_type);
                let n = param_slot_count(*param_type);
                let offset = if is_int {
                    let o = int_offset;
                    int_offset += n;
                    o
                } else {
                    let o = float_offset;
                    float_offset += n;
                    o
                };
                param_slots[i] = Some(ParamSlot {
                    offset,
                    is_int,
                    num_vec4s: n,
                });
            }
            PixelBenderParam::Texture { index, .. } => {
                inputs.push(*index);
            }
        }
    }

    let (output_reg, _output_type) = shader
        .output_reg()
        .ok_or_else(|| "PixelBender shader has no output parameter".to_string())?;
    mark(output_reg, &mut used_float, &mut used_int);
    let output_channels = output_reg.channels.len();

    // Collect registers touched by operations, and which inputs are sampled
    // with nearest filtering (ES 1.00 filtering is per-texture, not per-sample).
    let mut nearest_tfs: BTreeSet<u8> = BTreeSet::new();
    for op in &shader.operations {
        match op {
            Operation::Normal { dst, src, .. } => {
                mark(dst, &mut used_float, &mut used_int);
                mark(src, &mut used_float, &mut used_int);
            }
            Operation::LoadInt { dst, .. } | Operation::LoadFloat { dst, .. } => {
                mark(dst, &mut used_float, &mut used_int);
            }
            Operation::If { src } => mark(src, &mut used_float, &mut used_int),
            Operation::SampleNearest { dst, src, tf } => {
                mark(dst, &mut used_float, &mut used_int);
                mark(src, &mut used_float, &mut used_int);
                nearest_tfs.insert(*tf);
            }
            Operation::SampleLinear { dst, src, .. } => {
                mark(dst, &mut used_float, &mut used_int);
                mark(src, &mut used_float, &mut used_int);
            }
            Operation::Select {
                src1,
                src2,
                condition,
                dst,
            } => {
                mark(src1, &mut used_float, &mut used_int);
                mark(src2, &mut used_float, &mut used_int);
                mark(condition, &mut used_float, &mut used_int);
                mark(dst, &mut used_float, &mut used_int);
            }
            Operation::Nop | Operation::Else | Operation::EndIf => {}
        }
    }
    // Comparisons force-write int register 0.
    used_int.insert(0);

    let float_slots = float_offset.max(1);
    let int_slots = int_offset.max(1);

    // ---- Emit GLSL (body only; the target preamble is prepended at compile). ----
    let mut b = String::new();
    let _ = writeln!(b, "uniform vec4 u_float_params[{float_slots}];");
    let _ = writeln!(b, "uniform ivec4 u_int_params[{int_slots}];");
    let _ = writeln!(b, "uniform int u_zeroed;");
    let _ = writeln!(b, "uniform vec2 u_out_size;");
    for &idx in &inputs {
        let _ = writeln!(b, "uniform sampler2D u_tex_{idx};");
        let _ = writeln!(b, "uniform vec2 u_tex_size_{idx};");
    }
    // float->int uses round-half-to-even (matches naga's `round` / WGSL `round`,
    // which ties to the nearest even whole number). GLSL ES 1.00 has no
    // `roundEven`, so emit it component-wise.
    let _ = writeln!(b, "vec4 pbRoundEven(vec4 x) {{");
    let _ = writeln!(b, "    vec4 r = floor(x + 0.5);");
    let _ = writeln!(b, "    vec4 tie = vec4(equal(abs(r - x), vec4(0.5)));");
    let _ = writeln!(b, "    vec4 even = 2.0 * floor(r * 0.5);");
    let _ = writeln!(b, "    return mix(r, even, tie);");
    let _ = writeln!(b, "}}");
    let _ = writeln!(b, "void main() {{");

    for i in &used_float {
        let _ = writeln!(b, "    vec4 ff{i} = vec4(0.0);");
    }
    for i in &used_int {
        let _ = writeln!(b, "    ivec4 ii{i} = ivec4(0);");
    }

    // Load parameters into their registers.
    for (i, param) in shader.params.iter().enumerate() {
        if let (
            Some(slot),
            PixelBenderParam::Normal {
                reg, param_type, ..
            },
        ) = (param_slots[i], param)
        {
            emit_param_load(&mut b, reg, *param_type, slot);
        }
    }

    // outCoord = gl_FragCoord (pixel coordinates, top-left origin).
    if let Some(oc) = &out_coord {
        let mask: String = oc.channels.iter().map(|&c| chan_letter(c)).collect();
        let mask = if mask.is_empty() { "x".to_string() } else { mask };
        let k = oc.channels.len().max(1);
        // Use gl_FragCoord directly (like the naga port): rendering into an FBO
        // texture and sampling it later share the same row-0 origin, so no flip
        // is needed — flipping here would mirror spatial effects (dither, blur).
        let coord = "gl_FragCoord.xy";
        let src_sw: String = XYZW[..k].iter().collect();
        let _ = writeln!(b, "    vec4 _oc = vec4({coord}, 0.0, 0.0);");
        let _ = writeln!(b, "    {}.{} = _oc.{};", reg_name(oc), mask, src_sw);
    }

    // Operations.
    let mut indent = 1usize;
    for op in &shader.operations {
        emit_op(&mut b, op, &mut indent);
    }

    // Output. For < 4 channels, alpha is forced to 1.0 (matching the naga port).
    if output_channels < 4 {
        let _ = writeln!(
            b,
            "    gl_FragColor = vec4(({}).xyz, 1.0);",
            load_padded(output_reg)
        );
    } else {
        let _ = writeln!(b, "    gl_FragColor = {};", load_padded(output_reg));
    }
    let _ = writeln!(b, "}}");

    Ok(TranslatedShader {
        glsl: b,
        float_slots,
        int_slots,
        inputs,
        nearest_inputs: nearest_tfs.into_iter().collect(),
        param_slots,
        output_channels,
    })
}

fn emit_param_load(
    b: &mut String,
    reg: &PixelBenderReg,
    ty: PixelBenderTypeOpcode,
    slot: ParamSlot,
) {
    let arr = if slot.is_int {
        "u_int_params"
    } else {
        "u_float_params"
    };
    match ty {
        PixelBenderTypeOpcode::TFloat2x2 => {
            // Packed in one vec4: col0 = xy, col1 = zw. Stored into one register.
            let _ = writeln!(
                b,
                "    ff{}.xyzw = {arr}[{}].xyzw;",
                reg.index, slot.offset
            );
        }
        PixelBenderTypeOpcode::TFloat3x3 => {
            for c in 0..3 {
                let _ = writeln!(
                    b,
                    "    ff{}.xyz = {arr}[{}].xyz;",
                    reg.index + c as u32,
                    slot.offset + c
                );
            }
        }
        PixelBenderTypeOpcode::TFloat4x4 => {
            for c in 0..4 {
                let _ = writeln!(
                    b,
                    "    ff{}.xyzw = {arr}[{}].xyzw;",
                    reg.index + c as u32,
                    slot.offset + c
                );
            }
        }
        _ => {
            // Scalar/vector: load the leading components into the reg's channels.
            let mask: String = reg.channels.iter().map(|&c| chan_letter(c)).collect();
            let mask = if mask.is_empty() { "x".to_string() } else { mask };
            let k = reg.channels.len().max(1);
            let src_sw: String = XYZW[..k].iter().collect();
            let _ = writeln!(
                b,
                "    {}.{} = {arr}[{}].{};",
                reg_name(reg),
                mask,
                slot.offset,
                src_sw
            );
        }
    }
}

fn pad(indent: usize) -> String {
    "    ".repeat(indent)
}

fn emit_op(b: &mut String, op: &Operation, indent: &mut usize) {
    match op {
        Operation::Nop => {}
        Operation::LoadFloat { dst, val } => {
            store(b, dst, &format!("vec4({val:?})"));
        }
        Operation::LoadInt { dst, val } => {
            store(b, dst, &format!("ivec4({val})"));
        }
        Operation::If { src } => {
            let cond = load_exact(src);
            let test = match src.kind {
                PixelBenderRegKind::Float => format!("({cond}) != 0.0"),
                PixelBenderRegKind::Int => format!("({cond}) != 0"),
            };
            let _ = writeln!(b, "{}if ({test}) {{", pad(*indent));
            *indent += 1;
        }
        Operation::Else => {
            *indent = indent.saturating_sub(1);
            let _ = writeln!(b, "{}}} else {{", pad(*indent));
            *indent += 1;
        }
        Operation::EndIf => {
            *indent = indent.saturating_sub(1);
            let _ = writeln!(b, "{}}}", pad(*indent));
        }
        Operation::Select {
            src1,
            src2,
            condition,
            dst,
        } => {
            let c = load_exact(condition);
            let test = match condition.kind {
                PixelBenderRegKind::Float => format!("({c}) == 1.0"),
                PixelBenderRegKind::Int => format!("({c}) == 1"),
            };
            let value = format!(
                "({test}) ? {} : {}",
                load_padded(src1),
                load_padded(src2)
            );
            store(b, dst, &value);
        }
        Operation::SampleNearest { dst, src, tf }
        | Operation::SampleLinear { dst, src, tf } => {
            let coord = load_exact(src); // vec2 pixel coords
            let uv = format!("(vec2({coord}) / u_tex_size_{tf})");
            let sample = format!("texture2D(u_tex_{tf}, {uv})");
            // Out-of-range zeroing (Filter mode): transparent black outside [0,1].
            let value = format!(
                "((u_zeroed != 0 && ({uv}.x < 0.0 || {uv}.x > 1.0 || {uv}.y < 0.0 || {uv}.y > 1.0)) ? vec4(0.0) : {sample})"
            );
            store(b, dst, &value);
        }
        Operation::Normal { opcode, dst, src } => {
            emit_normal(b, *opcode, dst, src);
        }
    }
}

/// Pads a `vecN` expression to `vec4`. `n == 1` splats (matching length/dot,
/// which the naga port broadcasts); `n == 2/3` zero-pads; `n == 4` is a no-op.
fn pad_to_vec4(expr: String, n: usize) -> String {
    match n {
        0 | 1 => format!("vec4({expr})"),
        2 => format!("vec4(({expr}), 0.0, 0.0)"),
        3 => format!("vec4(({expr}), 0.0)"),
        _ => expr,
    }
}

/// Builds a `matN` from the register span a matrix channel occupies
/// (column-major: M2x2 packed in one reg, M3x3/M4x4 in 3/4 consecutive regs).
fn matrix_expr(reg: &PixelBenderReg) -> (String, usize) {
    let i = reg.index;
    match reg.channels.first() {
        Some(PixelBenderRegChannel::M2x2) => (format!("mat2(ff{i}.xy, ff{i}.zw)"), 2),
        Some(PixelBenderRegChannel::M3x3) => (
            format!("mat3(ff{i}.xyz, ff{}.xyz, ff{}.xyz)", i + 1, i + 2),
            3,
        ),
        Some(PixelBenderRegChannel::M4x4) => (
            format!("mat4(ff{i}, ff{}, ff{}, ff{})", i + 1, i + 2, i + 3),
            4,
        ),
        _ => (format!("mat2(ff{i}.xy, ff{i}.zw)"), 2),
    }
}

/// True if the register's first channel is a matrix channel (M2x2/M3x3/M4x4),
/// so the register spans a whole matrix rather than named vector lanes.
fn is_matrix_channel(reg: &PixelBenderReg) -> bool {
    matches!(
        reg.channels.first(),
        Some(
            PixelBenderRegChannel::M2x2
                | PixelBenderRegChannel::M3x3
                | PixelBenderRegChannel::M4x4
        )
    )
}

fn emit_normal(b: &mut String, opcode: Opcode, dst: &PixelBenderReg, src: &PixelBenderReg) {
    use Opcode::*;
    // A matrix-typed destination (e.g. `mat = mat`) must copy the entire matrix
    // span, not a single lane — the swizzle machinery below only names x/y/z/w.
    if is_matrix_channel(dst) {
        let (src_expr, n) = matrix_expr(src);
        match opcode {
            Mov => emit_matrix_store(b, dst, &src_expr, n),
            _ => {
                // Elementwise matrix op: `dst <op> src` per column.
                let (dst_expr, _) = matrix_expr(dst);
                let op = match opcode {
                    Add => "+",
                    Sub => "-",
                    Mul => "*",
                    Div => "/",
                    _ => "+",
                };
                emit_matrix_store(b, dst, &format!("({dst_expr} {op} {src_expr})"), n);
            }
        }
        return;
    }
    let s = load_padded(src);
    let d = load_padded(dst);
    let sn = src.channels.len().max(1);
    // Binary ops use dst as the first operand.
    let bin = |op: &str| format!("({d}) {op} ({s})");
    let un = |f: &str| format!("{f}({s})");

    let value: String = match opcode {
        Mov => s.clone(),
        Rcp => format!("vec4(1.0) / ({s})"),
        Add => bin("+"),
        Sub => bin("-"),
        Mul => bin("*"),
        Div => bin("/"),
        Min => format!("min({s}, {d})"),
        Max => format!("max({s}, {d})"),
        Step => format!("step({s}, {d})"),
        Pow => format!("pow({d}, {s})"),
        Mod => format!("mod({d}, {s})"),
        Atan2 => format!("atan({d}, {s})"),
        Abs => un("abs"),
        Sign => un("sign"),
        Floor => un("floor"),
        Ceil => un("ceil"),
        Fract => un("fract"),
        Sin => un("sin"),
        Cos => un("cos"),
        Tan => un("tan"),
        Asin => un("asin"),
        Acos => un("acos"),
        Atan => un("atan"),
        Exp => un("exp"),
        Exp2 => un("exp2"),
        Log => un("log"),
        Log2 => un("log2"),
        Sqrt => pad_to_vec4(format!("sqrt({})", load_exact(src)), sn),
        RSqrt => pad_to_vec4(format!("inversesqrt({})", load_exact(src)), sn),
        Normalize => pad_to_vec4(format!("normalize({})", load_exact(src)), sn),
        Length => format!("vec4(length({}))", load_exact(src)),
        Distance => format!("vec4(distance({}, {}))", load_exact(dst), load_exact(src)),
        DotProduct => format!("vec4(dot({}, {}))", load_exact(dst), load_exact(src)),
        CrossProduct => pad_to_vec4(
            format!("cross({}, {})", load_exact(dst), load_exact(src)),
            3,
        ),
        FloatToInt => format!("pbRoundEven({s})"),
        IntToFloat => s.clone(),
        FloatToBool => format!("vec4(notEqual(({s}), vec4(0.0)))"),
        BoolToFloat => "vec4(0.0)".to_string(),
        IntToBool => s.clone(),
        LogicalNot => format!("(ivec4(-1) - ({s}))"),
        LogicalAnd => {
            format!("vec4(vec4(notEqual({d}, vec4(0.0))) * vec4(notEqual({s}, vec4(0.0))))")
        }
        LogicalOr => format!(
            "(vec4(1.0) - vec4(vec4(equal({d}, vec4(0.0))) * vec4(equal({s}, vec4(0.0)))))"
        ),
        LogicalXor => {
            format!("abs(vec4(notEqual({d}, vec4(0.0))) - vec4(notEqual({s}, vec4(0.0))))")
        }
        // Comparisons: result forced into int register 0, channel R.
        Equal | NotEqual | LessThan | LessThanEqual => {
            let op = match opcode {
                Equal => "==",
                NotEqual => "!=",
                LessThan => "<",
                LessThanEqual => "<=",
                _ => unreachable!(),
            };
            let _ = writeln!(
                b,
                "    ii0.x = (({}) {op} ({})) ? 1 : 0;",
                load_exact(dst),
                load_exact(src)
            );
            return;
        }
        MatVecMul => {
            // dst is a vector, src the matrix: matrix * vector.
            let (m, n) = matrix_expr(src);
            let v = format!(
                "{}.{}",
                reg_name(dst),
                &"xyzw"[..n]
            );
            pad_to_vec4(format!("({m} * {v})"), n)
        }
        VecMatMul => {
            // dst is a vector, src the matrix: vector * matrix.
            let (m, n) = matrix_expr(src);
            let v = format!("{}.{}", reg_name(dst), &"xyzw"[..n]);
            pad_to_vec4(format!("({v} * {m})"), n)
        }
        MatMatMul => {
            let (md, _) = matrix_expr(dst);
            let (ms, n) = matrix_expr(src);
            // Store the product columns back below; fall through to a plain store
            // isn't meaningful for matrices, so emit column stores directly.
            let prod = format!("({md} * {ms})");
            emit_matrix_store(b, dst, &prod, n);
            return;
        }
        _ => s.clone(),
    };

    store(b, dst, &value);
}

/// Stores a `matN` product back into the destination's register span.
fn emit_matrix_store(b: &mut String, dst: &PixelBenderReg, mat_expr: &str, n: usize) {
    let i = dst.index;
    let _ = writeln!(b, "    {{ mat{n} _m = {mat_expr};");
    match n {
        2 => {
            let _ = writeln!(b, "    ff{i}.xy = _m[0]; ff{i}.zw = _m[1]; }}");
        }
        3 => {
            let _ = writeln!(
                b,
                "    ff{i}.xyz = _m[0]; ff{}.xyz = _m[1]; ff{}.xyz = _m[2]; }}",
                i + 1,
                i + 2
            );
        }
        _ => {
            let _ = writeln!(
                b,
                "    ff{i} = _m[0]; ff{} = _m[1]; ff{} = _m[2]; ff{} = _m[3]; }}",
                i + 1,
                i + 2,
                i + 3
            );
        }
    }
}
