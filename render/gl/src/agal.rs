//! Translates Adobe AGAL (Stage3D shader bytecode) into GLSL ES 1.00, so the
//! native-GL Stage3D backend can run on the same oldest-GL targets as the rest of
//! the renderer (naga's GLSL backend only emits ES 3.00+).
//!
//! AGAL registers are all `vec4`; the model mirrors the PixelBender translator:
//! sources carry a 4-component swizzle, destinations a write mask, and each opcode
//! becomes a masked assignment. Vertex and fragment programs are translated
//! together so their `varying` declarations match.

use naga_agal::{
    DestField, DirectMode, Mask, Opcode, ParsedBytecode, RegisterType, ShaderType, Source2,
    SourceField,
};
use std::collections::BTreeSet;
use std::fmt::Write;

/// Sampler used by a fragment program.
#[derive(Clone, Copy)]
pub struct SamplerInfo {
    pub reg: u16,
    pub cube: bool,
}

pub struct TranslatedProgram {
    pub vertex_glsl: String,
    pub fragment_glsl: String,
    /// Number of `vec4` slots the shader indexes in the constant array.
    pub num_vertex_constants: usize,
    pub num_fragment_constants: usize,
    /// Vertex-attribute register numbers the vertex program reads (→ `va{n}`).
    pub attributes: Vec<u16>,
    pub samplers: Vec<SamplerInfo>,
}

const XYZW: [char; 4] = ['x', 'y', 'z', 'w'];

fn swizzle_str(swizzle: u8) -> String {
    (0..4)
        .map(|i| XYZW[((swizzle >> (2 * i)) & 3) as usize])
        .collect()
}

fn mask_str(mask: Mask) -> String {
    let mut s = String::new();
    if mask.contains(Mask::X) {
        s.push('x');
    }
    if mask.contains(Mask::Y) {
        s.push('y');
    }
    if mask.contains(Mask::Z) {
        s.push('z');
    }
    if mask.contains(Mask::W) {
        s.push('w');
    }
    if s.is_empty() {
        s.push_str("xyzw");
    }
    s
}

fn const_prefix(stage: ShaderType) -> char {
    match stage {
        ShaderType::Vertex => 'v',
        ShaderType::Fragment => 'f',
    }
}

/// GLSL name (without swizzle) for a register.
fn reg_name(reg_type: &RegisterType, reg_num: u16, stage: ShaderType) -> String {
    let p = const_prefix(stage);
    match reg_type {
        RegisterType::Attribute => format!("va{reg_num}"),
        RegisterType::Constant => format!("{p}c[{reg_num}]"),
        RegisterType::Temporary => format!("{p}t{reg_num}"),
        RegisterType::Varying => format!("v{reg_num}"),
        RegisterType::Sampler => format!("fs{reg_num}"),
        RegisterType::Output => match stage {
            ShaderType::Vertex => "_agal_out_pos".to_string(),
            ShaderType::Fragment => "_agal_out_col".to_string(),
        },
        RegisterType::FragmentRegister => "gl_FragCoord".to_string(),
    }
}

/// A `vec4` rvalue for a source operand (base register with its swizzle applied).
///
/// Indirect operands (`vc[index_reg.c + offset]`, only into the constant array)
/// dynamically index the uniform array by a component of another register.
fn source_ref(s: &SourceField, stage: ShaderType) -> String {
    if matches!(s.direct_mode, DirectMode::Indirect) {
        let index_reg = reg_name(&s.index_type, s.reg_num, stage);
        let comp = XYZW[s.index_select as usize & 3];
        let p = const_prefix(stage);
        return format!(
            "{p}c[int({index_reg}.{comp}) + {}].{}",
            s.indirect_offset,
            swizzle_str(s.swizzle)
        );
    }
    format!("{}.{}", reg_name(&s.register_type, s.reg_num, stage), swizzle_str(s.swizzle))
}

fn dest_lvalue(d: &DestField, stage: ShaderType) -> String {
    reg_name(&d.register_type, d.reg_num, stage)
}

/// Emits `dst.<mask> = (rhs).<mask>;` (rhs must be a `vec4` expression).
fn store(out: &mut String, dst: &DestField, rhs: &str, stage: ShaderType) {
    let lv = dest_lvalue(dst, stage);
    let mask = mask_str(dst.write_mask);
    let _ = writeln!(out, "    {lv}.{mask} = ({rhs}).{mask};");
}

/// Consecutive matrix-row registers `base .. base+rows` (no swizzle).
fn matrix_rows(base: &SourceField, rows: u16, stage: ShaderType) -> Vec<String> {
    (0..rows)
        .map(|i| reg_name(&base.register_type, base.reg_num + i, stage))
        .collect()
}

/// Tracks which registers/varyings/samplers a program touches, per type.
#[derive(Default)]
struct Usage {
    attributes: BTreeSet<u16>,
    constants: BTreeSet<u16>,
    temporaries: BTreeSet<u16>,
    varyings: BTreeSet<u16>,
    samplers: BTreeSet<(u16, bool)>,
    uses_output: bool,
    /// Set when a constant is indexed indirectly, so the whole uploaded constant
    /// bank must be declared (the runtime index isn't known at translation time).
    uses_indirect: bool,
}

impl Usage {
    fn mark_source(&mut self, s: &SourceField) {
        if matches!(s.direct_mode, DirectMode::Indirect) {
            self.uses_indirect = true;
            // Mark the index register (e.g. the vertex attribute selecting a color).
            match s.index_type {
                RegisterType::Attribute => {
                    self.attributes.insert(s.reg_num);
                }
                RegisterType::Temporary => {
                    self.temporaries.insert(s.reg_num);
                }
                _ => {}
            }
            return;
        }
        match s.register_type {
            RegisterType::Attribute => {
                self.attributes.insert(s.reg_num);
            }
            RegisterType::Constant => {
                self.constants.insert(s.reg_num);
            }
            RegisterType::Temporary => {
                self.temporaries.insert(s.reg_num);
            }
            RegisterType::Varying => {
                self.varyings.insert(s.reg_num);
            }
            _ => {}
        }
    }

    fn mark_dest(&mut self, d: &DestField) {
        match d.register_type {
            RegisterType::Temporary => {
                self.temporaries.insert(d.reg_num);
            }
            RegisterType::Varying => {
                self.varyings.insert(d.reg_num);
            }
            RegisterType::Output => self.uses_output = true,
            _ => {}
        }
    }
}

/// Number of `vec4` slots the constant uniform array must declare. Indirect
/// indexing forces the full AGAL bank, since the runtime index is unknown here.
fn const_count(usage: &Usage, stage: ShaderType) -> usize {
    if usage.uses_indirect {
        match stage {
            ShaderType::Vertex => 128,
            ShaderType::Fragment => 28,
        }
    } else {
        usage
            .constants
            .iter()
            .max()
            .map(|m| *m as usize + 1)
            .unwrap_or(0)
    }
}

fn scan(parsed: &ParsedBytecode, stage: ShaderType) -> Usage {
    let mut u = Usage::default();
    for (opcode, dst, s1, s2) in parsed.operations() {
        if !matches!(
            opcode,
            Opcode::Kil | Opcode::Ife | Opcode::Ine | Opcode::Ifg | Opcode::Ifl
        ) {
            u.mark_dest(dst);
        }
        u.mark_source(s1);
        match s2 {
            Source2::SourceField(s) => {
                // Matrix opcodes read consecutive rows starting at s2.
                let rows = match opcode {
                    Opcode::M33 | Opcode::M34 => 3,
                    Opcode::M44 => 4,
                    _ => 1,
                };
                for i in 0..rows {
                    if s.register_type == RegisterType::Constant {
                        u.constants.insert(s.reg_num + i);
                    } else if s.register_type == RegisterType::Temporary {
                        u.temporaries.insert(s.reg_num + i);
                    }
                }
                u.mark_source(s);
            }
            Source2::Sampler(sampler) => {
                let cube = matches!(sampler.dimension, naga_agal::Dimension::Cube);
                u.samplers.insert((sampler.reg_num, cube));
            }
        }
    }
    let _ = stage;
    u
}

/// Emits one opcode as GLSL statements into `body`.
fn emit_op(
    body: &mut String,
    opcode: &Opcode,
    dst: &DestField,
    s1: &SourceField,
    s2: &Source2,
    stage: ShaderType,
    indent: &mut usize,
) {
    let a = source_ref(s1, stage);
    let b = match s2 {
        Source2::SourceField(s) => source_ref(s, stage),
        Source2::Sampler(_) => String::new(),
    };
    let pad = "    ".repeat(*indent);

    // Binary/unary opcodes that reduce to a single vec4 store.
    let rhs: Option<String> = match opcode {
        Opcode::Mov => Some(a.clone()),
        Opcode::Add => Some(format!("({a}) + ({b})")),
        Opcode::Sub => Some(format!("({a}) - ({b})")),
        Opcode::Mul => Some(format!("({a}) * ({b})")),
        Opcode::Div => Some(format!("({a}) / ({b})")),
        Opcode::Rcp => Some(format!("vec4(1.0) / ({a})")),
        Opcode::Min => Some(format!("min({a}, {b})")),
        Opcode::Max => Some(format!("max({a}, {b})")),
        Opcode::Frc => Some(format!("fract({a})")),
        Opcode::Sqt => Some(format!("sqrt({a})")),
        Opcode::Rsq => Some(format!("inversesqrt({a})")),
        Opcode::Pow => Some(format!("pow({a}, {b})")),
        Opcode::Log => Some(format!("log2({a})")),
        Opcode::Exp => Some(format!("exp2({a})")),
        Opcode::Nrm => Some(format!("vec4(normalize(({a}).xyz), 0.0)")),
        Opcode::Sin => Some(format!("sin({a})")),
        Opcode::Cos => Some(format!("cos({a})")),
        Opcode::Crs => Some(format!("vec4(cross(({a}).xyz, ({b}).xyz), 0.0)")),
        Opcode::Dp3 => Some(format!("vec4(dot(({a}).xyz, ({b}).xyz))")),
        Opcode::Dp4 => Some(format!("vec4(dot({a}, {b}))")),
        Opcode::Abs => Some(format!("abs({a})")),
        Opcode::Neg => Some(format!("-({a})")),
        Opcode::Sat => Some(format!("clamp({a}, 0.0, 1.0)")),
        Opcode::Sge => Some(format!("vec4(greaterThanEqual({a}, {b}))")),
        Opcode::Slt => Some(format!("vec4(lessThan({a}, {b}))")),
        Opcode::Seq => Some(format!("vec4(equal({a}, {b}))")),
        Opcode::Sne => Some(format!("vec4(notEqual({a}, {b}))")),
        Opcode::Ddx => Some(format!("dFdx({a})")),
        Opcode::Ddy => Some(format!("dFdy({a})")),
        _ => None,
    };
    if let Some(rhs) = rhs {
        body.push_str(&pad);
        store(body, dst, &rhs, stage);
        return;
    }

    match opcode {
        Opcode::M44 | Opcode::M34 | Opcode::M33 => {
            let s2f = match s2 {
                Source2::SourceField(s) => s,
                _ => return,
            };
            let (rows, comps, pad_tail) = match opcode {
                Opcode::M44 => (4, "", ""),
                Opcode::M34 => (3, "", ", 0.0"),
                _ => (3, ".xyz", ", 0.0"),
            };
            let m = matrix_rows(s2f, rows, stage);
            let v = &a;
            let dots: Vec<String> = m
                .iter()
                .map(|r| format!("dot(({r}){comps}, ({v}){comps})"))
                .collect();
            let rhs = format!("vec4({}{pad_tail})", dots.join(", "));
            body.push_str(&pad);
            store(body, dst, &rhs, stage);
        }
        Opcode::Tex => {
            if let Source2::Sampler(sampler) = s2 {
                let samp = format!("fs{}", sampler.reg_num);
                let rhs = if matches!(sampler.dimension, naga_agal::Dimension::Cube) {
                    format!("textureCube({samp}, ({a}).xyz)")
                } else {
                    format!("texture2D({samp}, ({a}).xy)")
                };
                body.push_str(&pad);
                store(body, dst, &rhs, stage);
            }
        }
        Opcode::Kil => {
            let _ = writeln!(body, "{pad}if (({a}).x < 0.0) discard;");
        }
        Opcode::Ife | Opcode::Ine | Opcode::Ifg | Opcode::Ifl => {
            let op = match opcode {
                Opcode::Ife => "==",
                Opcode::Ine => "!=",
                Opcode::Ifg => ">",
                Opcode::Ifl => "<",
                _ => unreachable!(),
            };
            let _ = writeln!(body, "{pad}if (({a}).x {op} ({b}).x) {{");
            *indent += 1;
        }
        Opcode::Els => {
            *indent = indent.saturating_sub(1);
            let p = "    ".repeat(*indent);
            let _ = writeln!(body, "{p}}} else {{");
            *indent += 1;
        }
        Opcode::Eif => {
            *indent = indent.saturating_sub(1);
            let p = "    ".repeat(*indent);
            let _ = writeln!(body, "{p}}}");
        }
        _ => {}
    }
}

fn emit_shader(parsed: &ParsedBytecode, stage: ShaderType, varyings: &BTreeSet<u16>) -> (String, Usage) {
    let usage = scan(parsed, stage);
    let mut b = String::new();

    match stage {
        ShaderType::Vertex => {
            for &a in &usage.attributes {
                let _ = writeln!(b, "attribute vec4 va{a};");
            }
            let nc = const_count(&usage, stage);
            if nc > 0 {
                let _ = writeln!(b, "uniform vec4 vc[{nc}];");
            }
            for &v in varyings {
                let _ = writeln!(b, "varying vec4 v{v};");
            }
        }
        ShaderType::Fragment => {
            for &v in varyings {
                let _ = writeln!(b, "varying vec4 v{v};");
            }
            let nc = const_count(&usage, stage);
            if nc > 0 {
                let _ = writeln!(b, "uniform vec4 fc[{nc}];");
            }
            for &(s, cube) in &usage.samplers {
                if cube {
                    let _ = writeln!(b, "uniform samplerCube fs{s};");
                } else {
                    let _ = writeln!(b, "uniform sampler2D fs{s};");
                }
            }
        }
    }

    let _ = writeln!(b, "void main() {{");
    for &t in &usage.temporaries {
        let p = const_prefix(stage);
        let _ = writeln!(b, "    vec4 {p}t{t} = vec4(0.0);");
    }
    if usage.uses_output {
        let name = match stage {
            ShaderType::Vertex => "_agal_out_pos",
            ShaderType::Fragment => "_agal_out_col",
        };
        let _ = writeln!(b, "    vec4 {name} = vec4(0.0, 0.0, 0.0, 1.0);");
    }

    let mut indent = 1usize;
    for (opcode, dst, s1, s2) in parsed.operations() {
        emit_op(&mut b, opcode, dst, s1, s2, stage, &mut indent);
    }

    match stage {
        ShaderType::Vertex => {
            // Flash Stage3D uses D3D-style clip space (depth 0..1, +Y down relative
            // to GL). Map to GL: keep XY, remap Z from [0,w] to [-w,w].
            // Flash Stage3D uses D3D-style clip space (depth 0..1); remap Z to GL's
            // [-w, w]. Negate Y so the render-to-texture result is top-down, matching
            // how the stage composites the Stage3D bitmap.
            let _ = writeln!(
                b,
                "    gl_Position = vec4(_agal_out_pos.x, -_agal_out_pos.y, _agal_out_pos.z * 2.0 - _agal_out_pos.w, _agal_out_pos.w);"
            );
        }
        ShaderType::Fragment => {
            let _ = writeln!(b, "    gl_FragColor = _agal_out_col;");
        }
    }
    let _ = writeln!(b, "}}");
    (b, usage)
}

/// Translates a vertex/fragment AGAL pair into a GLSL ES 1.00 program.
pub fn translate(
    vertex: &ParsedBytecode,
    fragment: &ParsedBytecode,
) -> Result<TranslatedProgram, String> {
    if !matches!(vertex.shader_type(), ShaderType::Vertex) {
        return Err("expected a vertex program".into());
    }
    if !matches!(fragment.shader_type(), ShaderType::Fragment) {
        return Err("expected a fragment program".into());
    }

    // Varyings must be declared identically in both stages.
    let v_usage = scan(vertex, ShaderType::Vertex);
    let f_usage = scan(fragment, ShaderType::Fragment);
    let varyings: BTreeSet<u16> = v_usage.varyings.union(&f_usage.varyings).copied().collect();

    let (vertex_glsl, v_usage) = emit_shader(vertex, ShaderType::Vertex, &varyings);
    let (fragment_glsl, f_usage) = emit_shader(fragment, ShaderType::Fragment, &varyings);

    Ok(TranslatedProgram {
        vertex_glsl,
        fragment_glsl,
        num_vertex_constants: const_count(&v_usage, ShaderType::Vertex),
        num_fragment_constants: const_count(&f_usage, ShaderType::Fragment),
        attributes: v_usage.attributes.iter().copied().collect(),
        samplers: f_usage
            .samplers
            .iter()
            .map(|&(reg, cube)| SamplerInfo { reg, cube })
            .collect(),
    })
}
