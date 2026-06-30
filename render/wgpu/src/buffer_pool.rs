use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};

type PoolInner<T> = Mutex<Vec<T>>;
type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

/// Upper bound on the bytes of *idle* (returned-to-pool) textures we keep cached.
/// When exceeded, least-recently-used size buckets are evicted.
///
/// The pool is keyed by exact texture size, and a key is created the first time
/// a given size is requested. Content rendered at continuously-varying sizes
/// (filters, `cacheAsBitmap`) would otherwise create one permanently-retained
/// texture per distinct size, growing without bound for as long as the content
/// keeps moving. LRU eviction caps that growth.
const TEXTURE_POOL_BYTE_BUDGET: u64 = 128 * 1024 * 1024;

#[derive(Debug, Default)]
pub struct TexturePool {
    pools: FnvHashMap<TextureKey, SizedPool>,
    globals_cache: FnvHashMap<GlobalsKey, Arc<Globals>>,
    /// Monotonic logical clock, bumped on every `get_texture`, used for LRU ordering.
    clock: u64,
}

#[derive(Debug)]
struct SizedPool {
    pool: BufferPool<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>,
    /// Estimated bytes of a single texture in this bucket.
    bytes_each: u64,
    /// `clock` value of the most recent `get_texture` for this size (for LRU).
    last_used: u64,
}

impl TexturePool {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_texture(
        &mut self,
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible> {
        let key = TextureKey {
            size,
            usage,
            format,
            sample_count,
        };
        self.clock += 1;
        let now = self.clock;
        let is_new = !self.pools.contains_key(&key);
        let slot = self.pools.entry(key).or_insert_with(|| {
            let label = if cfg!(feature = "render_debug_labels") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ID_COUNT: AtomicU32 = AtomicU32::new(0);
                let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
                create_debug_label!("Pooled texture {}", id)
            } else {
                None
            };
            SizedPool {
                pool: BufferPool::new(Box::new(move |descriptors, _description| {
                    let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
                        label: label.as_deref(),
                        size,
                        mip_level_count: 1,
                        sample_count,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        view_formats: &[format],
                        usage,
                    });
                    let view = texture.create_view(&Default::default());
                    (texture, view)
                })),
                bytes_each: texture_byte_size(&key),
                last_used: now,
            }
        });
        slot.last_used = now;
        let entry = slot.pool.take(descriptors, AlwaysCompatible);
        // Only a freshly-inserted size can push us over budget; checking here
        // (rather than every call) avoids locking every bucket on the hot path.
        // The just-used key has the newest `last_used`, so it is evicted last.
        if is_new {
            self.evict_if_needed();
        }
        entry
    }

    /// Evict least-recently-used size buckets until the cached (idle) texture
    /// bytes are back under [`TEXTURE_POOL_BYTE_BUDGET`].
    ///
    /// Only idle textures (sitting in a bucket's free list) count toward the
    /// budget; textures currently in flight are held by live `PoolEntry`s via a
    /// `Weak` handle, so evicting their bucket simply frees them on drop instead
    /// of returning them to the pool.
    fn evict_if_needed(&mut self) {
        let mut total: u64 = self
            .pools
            .values()
            .map(|s| s.bytes_each.saturating_mul(s.pool.available_len() as u64))
            .sum();
        if total <= TEXTURE_POOL_BYTE_BUDGET {
            return;
        }

        let mut by_age: Vec<(u64, TextureKey)> =
            self.pools.iter().map(|(k, s)| (s.last_used, *k)).collect();
        by_age.sort_unstable_by_key(|(last_used, _)| *last_used);

        for (_, key) in by_age {
            if total <= TEXTURE_POOL_BYTE_BUDGET {
                break;
            }
            if let Some(slot) = self.pools.remove(&key) {
                total = total
                    .saturating_sub(slot.bytes_each.saturating_mul(slot.pool.available_len() as u64));
            }
        }
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Arc<Globals> {
        self.globals_cache
            .entry(GlobalsKey {
                viewport_width,
                viewport_height,
            })
            .or_insert_with(|| {
                Arc::new(Globals::new(
                    &descriptors.device,
                    &descriptors.bind_layouts.globals,
                    viewport_width,
                    viewport_height,
                ))
            })
            .clone()
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct TextureKey {
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    sample_count: u32,
}

/// Estimated GPU memory of a single texture with the given key. Used only for
/// LRU budgeting, so an approximation (ignoring driver alignment/mip padding) is
/// fine. Compressed/odd formats with no defined copy size fall back to 4 bpp.
fn texture_byte_size(key: &TextureKey) -> u64 {
    let bytes_per_pixel = key.format.block_copy_size(None).unwrap_or(4) as u64;
    bytes_per_pixel
        .saturating_mul(key.size.width as u64)
        .saturating_mul(key.size.height as u64)
        .saturating_mul(key.size.depth_or_array_layers as u64)
        .saturating_mul(key.sample_count as u64)
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct GlobalsKey {
    viewport_width: u32,
    viewport_height: u32,
}

pub trait BufferDescription: Clone + Debug {
    type Cost: Ord;

    /// If the potential buffer represented by this description (`self`)
    /// fits another existing buffer and its description (`other`),
    /// return the cost to use that buffer instead of making a new one.
    ///
    /// Cost is an arbitrary unit, but lower is better.
    /// None means that the other buffer cannot be used in place of this one.
    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost>;
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
        Some(())
    }
}

pub struct BufferPool<Type, Description: BufferDescription> {
    available: Arc<PoolInner<(Type, Description)>>,
    constructor: Constructor<Type, Description>,
}

impl<Type, Description: BufferDescription> Debug for BufferPool<Type, Description> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish()
    }
}

impl<Type, Description: BufferDescription> BufferPool<Type, Description> {
    pub fn new(constructor: Constructor<Type, Description>) -> Self {
        Self {
            available: Arc::new(Mutex::new(vec![])),
            constructor,
        }
    }

    /// Number of idle buffers currently sitting in this pool's free list.
    pub fn available_len(&self) -> usize {
        self.available
            .lock()
            .expect("Should not be able to lock recursively")
            .len()
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self
            .available
            .lock()
            .expect("Should not be able to lock recursively");
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.len() {
            if let Some(cost) = description.cost_to_use(&guard[i].1) {
                if let Some(best) = &mut best {
                    if best.0 > cost {
                        *best = (cost, i);
                    }
                } else if best.is_none() {
                    best = Some((cost, i));
                }
            }
        }

        let (item, used_description) = if let Some((_, best)) = best {
            guard.swap_remove(best)
        } else {
            let item = (self.constructor)(descriptors, &description);
            (item, description)
        };
        PoolEntry {
            item: Some(item),
            description: used_description,
            pool: Arc::downgrade(&self.available),
        }
    }
}

pub struct PoolEntry<Type, Description: BufferDescription> {
    item: Option<Type>,
    description: Description,
    pool: Weak<PoolInner<(Type, Description)>>,
}

impl<Type, Description: BufferDescription> Debug for PoolEntry<Type, Description>
where
    Type: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PoolEntry").field(&self.item).finish()
    }
}

impl<Type, Description: BufferDescription> Drop for PoolEntry<Type, Description> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take()
            && let Some(pool) = self.pool.upgrade()
        {
            pool.lock()
                .expect("Should not be able to lock recursively")
                .push((item, self.description.clone()))
        }
    }
}

impl<Type, Description: BufferDescription> Deref for PoolEntry<Type, Description> {
    type Target = Type;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Item should exist until dropped")
    }
}
