//! A size-keyed pool of reusable RGBA8 textures and shared framebuffer objects.
//!
//! Filters and (region-sized) blends render through many short-lived offscreen
//! textures and framebuffers. Creating and destroying GL objects on every pass
//! is wasteful — driver validation and allocation dominate. This mirrors the
//! wgpu backend's `TexturePool`/buffer pool: transient textures are recycled by
//! exact size under a byte budget, and the framebuffer objects used to attach
//! them are created once and re-attached rather than re-created.

use crate::context::GlContext;
use glow::HasContext as _;
use std::collections::HashMap;

/// Upper bound on the bytes of *idle* (returned-to-pool) textures we keep cached.
/// When exceeded, buckets are evicted until back under budget. Content rendered
/// at continuously-varying sizes (filters, moving blends) would otherwise retain
/// one texture per distinct size forever; the budget caps that growth.
const TEXTURE_POOL_BYTE_BUDGET: u64 = 64 * 1024 * 1024;

fn texture_bytes(width: u32, height: u32) -> u64 {
    (width as u64) * (height as u64) * 4
}

/// A pool of reusable RGBA8 textures keyed by exact `(width, height)`.
///
/// Acquired textures are removed from the pool and owned by the caller until
/// returned via [`TexturePool::release`]. Textures keep LINEAR filtering and
/// CLAMP_TO_EDGE wrapping; their contents are undefined on acquire, so callers
/// must fully overwrite (or explicitly clear) them before sampling.
pub(crate) struct TexturePool {
    gl: GlContext,
    free: HashMap<(u32, u32), Vec<glow::Texture>>,
    /// Total bytes of textures currently held in `free`.
    idle_bytes: u64,
}

impl TexturePool {
    pub(crate) fn new(gl: GlContext) -> Self {
        Self {
            gl,
            free: HashMap::new(),
            idle_bytes: 0,
        }
    }

    /// Returns a texture of exactly `width` x `height`, reusing an idle one if
    /// available or allocating a fresh RGBA8 texture otherwise. `None` only on
    /// allocation failure (context loss / OOM).
    pub(crate) fn acquire(&mut self, width: u32, height: u32) -> Option<glow::Texture> {
        if let Some(bucket) = self.free.get_mut(&(width, height)) {
            if let Some(texture) = bucket.pop() {
                self.idle_bytes = self.idle_bytes.saturating_sub(texture_bytes(width, height));
                return Some(texture);
            }
        }

        let gl = &self.gl;
        unsafe {
            let texture = gl.create_texture().ok()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA as i32,
                width as i32,
                height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            set_sampling(gl, glow::LINEAR);
            Some(texture)
        }
    }

    /// Returns a texture to the pool for reuse at its original size. Over-budget
    /// idle textures are evicted (deleted) to bound memory.
    pub(crate) fn release(&mut self, texture: glow::Texture, width: u32, height: u32) {
        self.free
            .entry((width, height))
            .or_default()
            .push(texture);
        self.idle_bytes += texture_bytes(width, height);
        self.evict_to_budget();
    }

    fn evict_to_budget(&mut self) {
        if self.idle_bytes <= TEXTURE_POOL_BYTE_BUDGET {
            return;
        }
        let gl = &self.gl;
        // Evict whole buckets until back under budget. Buckets for sizes no
        // longer in use are the natural first casualties as new sizes push the
        // total over budget.
        let mut sizes: Vec<(u32, u32)> = self.free.keys().copied().collect();
        while self.idle_bytes > TEXTURE_POOL_BYTE_BUDGET {
            let Some(size) = sizes.pop() else { break };
            if let Some(bucket) = self.free.remove(&size) {
                for texture in bucket {
                    unsafe { gl.delete_texture(texture) };
                    self.idle_bytes = self
                        .idle_bytes
                        .saturating_sub(texture_bytes(size.0, size.1));
                }
            }
        }
    }
}

impl Drop for TexturePool {
    fn drop(&mut self) {
        let gl = &self.gl;
        for bucket in self.free.values() {
            for &texture in bucket {
                unsafe { gl.delete_texture(texture) };
            }
        }
    }
}

/// Sets the standard filter-target sampling: LINEAR filtering, CLAMP_TO_EDGE
/// wrapping (the latter also keeps non-power-of-two targets valid on WebGL1).
pub(crate) unsafe fn set_sampling(gl: &glow::Context, filter: u32) {
    unsafe {
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter as i32);
    }
}
