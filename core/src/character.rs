use std::cell::{Cell, OnceCell, RefCell};
use std::rc::{Rc, Weak};

use crate::backend::audio::SoundHandle;
use crate::binary_data::BinaryData;
use crate::bitmap::bitmap_data::Color;
use crate::display_object::{
    Avm1Button, Avm2Button, BitmapClass, EditText, Graphic, MorphShape, MovieClip, Text, Video,
};
use crate::font::Font;
use gc_arena::barrier::unlock;
use gc_arena::lock::Lock;
use gc_arena::{Collect, Gc, Mutation};
use ruffle_render::backend::RenderBackend;
use ruffle_render::bitmap::{Bitmap as RenderBitmap, BitmapFormat, BitmapHandle, BitmapSize};
use ruffle_render::error::Error as RenderError;
use swf::DefineBitsLossless;

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub enum Character<'gc> {
    EditText(EditText<'gc>),
    Graphic(Graphic<'gc>),
    MovieClip(MovieClip<'gc>),
    Bitmap(Gc<'gc, BitmapCharacter<'gc>>),
    Avm1Button(Avm1Button<'gc>),
    Avm2Button(Avm2Button<'gc>),
    Font(Font<'gc>),
    MorphShape(MorphShape<'gc>),
    Text(Text<'gc>),
    Sound(#[collect(require_static)] SoundHandle),
    Video(Video<'gc>),
    BinaryData(BinaryData<'gc>),
}

#[derive(Collect, Debug)]
#[collect(no_drop)]
pub struct BitmapCharacter<'gc> {
    #[collect(require_static)]
    compressed: CompressedBitmap,
    /// A lazily constructed GPU handle, used when performing fills with this bitmap
    #[collect(require_static)]
    handle: OnceCell<BitmapHandle>,
    /// The bitmap class set by `SymbolClass` - this is used when we instantaite
    /// a `Bitmap` displayobject.
    avm2_class: Lock<BitmapClass<'gc>>,
    /// Weak reference to the most recently decoded CPU pixel buffer, shared
    /// (copy-on-write) across every `BitmapData` instantiated from this symbol.
    /// It's `Weak`, so once all instances drop the buffer is freed — a
    /// pathological SWF with thousands of distinct one-shot bitmaps never keeps
    /// them all decoded at once. `decoded_transparent` caches the format's
    /// transparency so a cache hit doesn't have to re-decode to learn it.
    #[collect(require_static)]
    decoded_pixels: RefCell<Weak<Vec<Color>>>,
    #[collect(require_static)]
    decoded_transparent: Cell<Option<bool>>,
}

impl<'gc> BitmapCharacter<'gc> {
    pub fn new(compressed: CompressedBitmap) -> Self {
        Self {
            compressed,
            handle: OnceCell::default(),
            avm2_class: Lock::new(BitmapClass::NoSubclass),
            decoded_pixels: RefCell::new(Weak::new()),
            decoded_transparent: Cell::new(None),
        }
    }

    pub fn compressed(&self) -> &CompressedBitmap {
        &self.compressed
    }

    /// Returns this symbol's decoded CPU pixels as a shared, copy-on-write
    /// buffer plus whether the source has an alpha channel. Instances built from
    /// the same symbol share one buffer (see `decoded_pixels`); the first call
    /// (or the first after all instances were dropped) decodes, later ones just
    /// clone the `Rc`.
    pub fn shared_pixels(&self) -> Result<(Rc<Vec<Color>>, bool), RenderError> {
        if let Some(pixels) = self.decoded_pixels.borrow().upgrade() {
            return Ok((pixels, self.decoded_transparent.get().unwrap_or(true)));
        }
        let decoded = self.compressed.decode()?;
        let transparent = matches!(decoded.format(), BitmapFormat::Rgba);
        let pixels: Rc<Vec<Color>> = Rc::new(decoded.as_colors().map(Color::from).collect());
        *self.decoded_pixels.borrow_mut() = Rc::downgrade(&pixels);
        self.decoded_transparent.set(Some(transparent));
        Ok((pixels, transparent))
    }

    pub fn avm2_class(&self) -> BitmapClass<'gc> {
        self.avm2_class.get()
    }

    pub fn set_avm2_class(this: Gc<'gc, Self>, bitmap_class: BitmapClass<'gc>, mc: &Mutation<'gc>) {
        unlock!(Gc::write(mc, this), Self, avm2_class).set(bitmap_class);
    }

    pub fn bitmap_handle(
        &self,
        backend: &mut dyn RenderBackend,
    ) -> Result<BitmapHandle, RenderError> {
        // FIXME - use `OnceCell::get_or_try_init` when stabilized.
        if let Some(handle) = self.handle.get() {
            return Ok(handle.clone());
        }
        let decoded = self.compressed.decode()?;
        let new_handle = backend.register_bitmap(decoded)?;
        // FIXME - do we ever want to release this handle, to avoid taking up GPU memory?
        self.handle.set(new_handle.clone()).unwrap();
        Ok(new_handle)
    }
}

/// Holds a bitmap from an SWF tag, plus the decoded width/height.
/// We avoid decompressing the image until it's actually needed - some pathological SWFS
/// like 'House' have thousands of highly-compressed (mostly empty) bitmaps, which can
/// take over 10GB of ram if we decompress them all during preloading.
#[derive(Clone, Debug)]
pub enum CompressedBitmap {
    Jpeg {
        data: Vec<u8>,
        alpha: Option<Vec<u8>>,
        width: u32,
        height: u32,
    },
    Lossless(DefineBitsLossless<'static>),
}

impl CompressedBitmap {
    pub fn size(&self) -> BitmapSize {
        match self {
            CompressedBitmap::Jpeg { width, height, .. } => BitmapSize {
                width: *width,
                height: *height,
            },
            CompressedBitmap::Lossless(define_bits_lossless) => BitmapSize {
                width: define_bits_lossless.width.into(),
                height: define_bits_lossless.height.into(),
            },
        }
    }
    pub fn decode(&self) -> Result<RenderBitmap<'static>, RenderError> {
        match self {
            CompressedBitmap::Jpeg {
                data,
                alpha,
                width: _,
                height: _,
            } => ruffle_render::utils::decode_define_bits_jpeg(data, alpha.as_deref()),
            CompressedBitmap::Lossless(define_bits_lossless) => {
                ruffle_render::utils::decode_define_bits_lossless(define_bits_lossless)
            }
        }
    }
}
