use ruffle_test_framework::environment::{CompileMode, Environment};

#[cfg(feature = "imgtests")]
use ruffle_render_wgpu::descriptors::Descriptors;
#[cfg(feature = "imgtests")]
use std::sync::{Arc, LazyLock};

pub struct NativeEnvironment {
    pub compile_mode: CompileMode,
    #[cfg(feature = "imgtests")]
    descriptors: LazyLock<Option<Arc<Descriptors>>>,
}

impl NativeEnvironment {
    pub fn new(compile_mode: CompileMode) -> Self {
        Self {
            compile_mode,
            #[cfg(feature = "imgtests")]
            descriptors: LazyLock::new(renderer::build_wgpu_descriptors),
        }
    }

    /// Wait for any pending GPU work to finish, with a `timeout`.
    ///
    /// Call this before `main` ends so the driver's worker threads quiesce
    /// while the environment is still alive — otherwise libc's
    /// `__cxa_finalize` can run while a Mesa lavapipe JIT thread is still
    /// compiling a shader, racing the destructors and producing a
    /// `SIGSEGV`. The wait also keeps wgpu's `Queue::Drop` (with its
    /// hard-coded ~6.3 s budget that `panic!`s on timeout) happy on slow
    /// software renderers like DX12-WARP on the Windows GitHub-Actions
    /// runner.
    pub fn flush_gpu_with_timeout(&self, #[allow(unused)] timeout: std::time::Duration) {
        #[cfg(feature = "imgtests")]
        {
            // `LazyLock::get` peeks without forcing initialization, so tests
            // that never touched the renderer don't pay the cost of building
            // wgpu state just to flush it again at shutdown.
            if let Some(Some(descriptors)) = LazyLock::get(&self.descriptors) {
                use ruffle_render_wgpu::wgpu;

                let _ = descriptors.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(timeout),
                });
            }
        }
    }
}

impl Environment for NativeEnvironment {
    #[cfg(feature = "imgtests")]
    fn is_render_supported(
        &self,
        _requirements: &ruffle_test_framework::options::RenderOptions,
    ) -> bool {
        // The native GL backend is available on demand; don't force-build wgpu.
        if std::env::var_os("RUFFLE_TEST_GL").is_some() {
            return true;
        }
        self.descriptors.is_some()
    }

    #[cfg(feature = "imgtests")]
    fn create_renderer(
        &self,
        width: u32,
        height: u32,
    ) -> Option<(
        Box<dyn ruffle_test_framework::environment::RenderInterface>,
        Box<dyn ruffle_test_framework::environment::RenderBackend>,
    )> {
        renderer::create_pair(self, width, height)
    }

    fn compile_mode(&self) -> CompileMode {
        self.compile_mode
    }
}

#[cfg(feature = "imgtests")]
mod renderer {
    use super::NativeEnvironment;
    use image::RgbaImage;
    use ruffle_render_wgpu::backend::{
        WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
    };
    use ruffle_render_wgpu::descriptors::Descriptors;
    use ruffle_render_wgpu::target::TextureTarget;
    use ruffle_render_wgpu::wgpu;
    use ruffle_test_framework::environment::{RenderBackend, RenderInterface};
    use std::any::Any;
    use std::sync::Arc;

    pub struct NativeRenderInterface {
        name: String,
    }

    pub fn create_pair(
        env: &NativeEnvironment,
        width: u32,
        height: u32,
    ) -> Option<(Box<dyn RenderInterface>, Box<dyn RenderBackend>)> {
        // Opt in to the native OpenGL backend for image tests.
        if std::env::var_os("RUFFLE_TEST_GL").is_some() {
            return create_gl_pair(width, height);
        }
        let descriptors = env.descriptors.clone()?;
        let target = TextureTarget::new(&descriptors.device, (width, height)).expect(
            "WGPU Texture Target creation must not fail, everything was checked ahead of time",
        );

        let adapter_info = descriptors.adapter.get_info();
        let name = format!("{}-{:?}", std::env::consts::OS, adapter_info.backend);

        Some((
            Box::new(NativeRenderInterface { name }),
            Box::new(WgpuRenderBackend::new(descriptors, target).expect(
                "WGPU Render backend creation must not fail, everything was checked ahead of time",
            )),
        ))
    }

    impl RenderInterface for NativeRenderInterface {
        fn name(&self) -> String {
            self.name.clone()
        }

        fn capture(&self, backend: &mut dyn RenderBackend) -> RgbaImage {
            let renderer =
                <dyn Any>::downcast_mut::<WgpuRenderBackend<TextureTarget>>(backend).unwrap();

            renderer.capture_frame().expect("Failed to capture image")
        }
    }

    // ---- Native OpenGL backend (headless via an EGL device + pbuffer) ----

    use glutin::api::egl::context::PossiblyCurrentContext;
    use glutin::api::egl::device::Device as EglDevice;
    use glutin::api::egl::display::Display as EglDisplay;
    use glutin::api::egl::surface::Surface as EglSurface;
    use glutin::config::ConfigTemplateBuilder;
    use glutin::context::{
        ContextApi, ContextAttributesBuilder, NotCurrentGlContext, Version,
    };
    use glutin::display::GlDisplay;
    use glutin::surface::{GlSurface, PbufferSurface, SurfaceAttributesBuilder};
    use ruffle_render::backend::ViewportDimensions;
    use ruffle_render::quality::StageQuality;
    use ruffle_render_gl::GlRenderBackend;
    use std::num::NonZeroU32;

    struct GlRenderInterface {
        // Kept alive (and current) for the lifetime of the backend.
        _context: PossiblyCurrentContext,
        _surface: EglSurface<PbufferSurface>,
        _display: EglDisplay,
    }

    impl RenderInterface for GlRenderInterface {
        fn name(&self) -> String {
            format!("{}-gl", std::env::consts::OS)
        }

        fn capture(&self, backend: &mut dyn RenderBackend) -> RgbaImage {
            let renderer = <dyn Any>::downcast_mut::<GlRenderBackend>(backend).unwrap();
            let (w, h, pixels) = renderer.read_framebuffer();
            RgbaImage::from_raw(w, h, pixels).expect("GL framebuffer readback failed")
        }
    }

    fn create_gl_pair(
        width: u32,
        height: u32,
    ) -> Option<(Box<dyn RenderInterface>, Box<dyn RenderBackend>)> {
        macro_rules! step {
            ($what:expr, $e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("GL test env: {} failed: {:?}", $what, e);
                        return None;
                    }
                }
            };
        }

        let devices: Vec<_> = step!("query_devices", EglDevice::query_devices()).collect();
        // `RUFFLE_TEST_GL_DEV=N` selects EGL device index N (e.g. a software
        // llvmpipe device); otherwise the first device is used.
        let device = match std::env::var("RUFFLE_TEST_GL_DEV")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(i) => devices.get(i),
            None => devices.first(),
        };
        let device = match device {
            Some(d) => d,
            None => {
                eprintln!("GL test env: no EGL devices found");
                return None;
            }
        };
        eprintln!("GL test env: using EGL device {:?}", device.name());
        let display = step!("with_device", unsafe {
            EglDisplay::with_device(&device, None)
        });

        let template = ConfigTemplateBuilder::new()
            .with_surface_type(glutin::config::ConfigSurfaceTypes::PBUFFER)
            .with_alpha_size(8)
            .build();
        let config = match step!("find_configs", unsafe { display.find_configs(template) }).next() {
            Some(c) => c,
            None => {
                eprintln!("GL test env: no EGL config found");
                return None;
            }
        };

        let ctx_attrs = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .build(None);
        let not_current = step!("create_context", unsafe {
            display.create_context(&config, &ctx_attrs)
        });

        let surf_attrs = SurfaceAttributesBuilder::<PbufferSurface>::new().build(
            NonZeroU32::new(width.max(1)).unwrap(),
            NonZeroU32::new(height.max(1)).unwrap(),
        );
        let surface = step!("create_pbuffer_surface", unsafe {
            display.create_pbuffer_surface(&config, &surf_attrs)
        });

        let context = step!("make_current", not_current.make_current(&surface));

        let mut backend = step!("GlRenderBackend::new", unsafe {
            GlRenderBackend::new_from_loader_function(
                |sym| {
                    let c = std::ffi::CString::new(sym).unwrap();
                    display.get_proc_address(c.as_c_str()) as *const _
                },
                false,
                // Match wgpu's test backend (`WgpuRenderBackend::new` builds its
                // surface at `StageQuality::Low`, i.e. no MSAA), so composited
                // bitmap edges are pixel-identical to the reference images.
                StageQuality::Low,
            )
        });
        backend.set_viewport_dimensions(ViewportDimensions {
            width,
            height,
            scale_factor: 1.0,
        });

        Some((
            Box::new(GlRenderInterface {
                _context: context,
                _surface: surface,
                _display: display,
            }),
            Box::new(backend),
        ))
    }

    /*
       It can be expensive to construct WGPU, much less Descriptors, so we put it off as long as we can
       and share it across tests in the same process.

       Remember:
           `cargo test` will run all tests in the same process.
           `cargo nextest run` will create a different process per test.

       For `cargo test` it's relatively okay if we spend the time to construct descriptors once,
       but for `cargo nextest run` it's a big cost per test if it's not going to use it.

       The descriptors live on the `NativeEnvironment` (rather than a `static`)
       so that they get dropped when the environment is dropped at the end of
       `main`. Their `Drop` impl runs `vkDestroyDevice` / `vkDestroyInstance`,
       which joins driver worker threads (notably Mesa lavapipe's LLVM JIT
       pool) before libc tears the process down. Leaving the descriptors in a
       `static` left those workers racing with `__cxa_finalize`, producing a
       `SIGSEGV` after the test had already passed.
    */

    fn create_wgpu_device() -> Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
        let instance = create_wgpu_instance(wgpu::Backends::all(), wgpu::BackendOptions::default());
        futures::executor::block_on(request_adapter_and_device(
            wgpu::Backends::all(),
            &instance,
            None,
            Default::default(),
        ))
        .ok()
        .map(|(adapter, device, queue)| (instance, adapter, device, queue))
    }

    pub(super) fn build_wgpu_descriptors() -> Option<Arc<Descriptors>> {
        let (instance, adapter, device, queue) = create_wgpu_device()?;

        Some(Arc::new(Descriptors::new(instance, adapter, device, queue)))
    }
}
