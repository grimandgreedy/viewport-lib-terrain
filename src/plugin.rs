//! `TerrainPlugin` implementation.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use viewport_lib::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickPassContext, PickRay,
    PluginItemCollection, SharedBindings, shared_wgsl,
};
use viewport_lib::renderer::PickId;
use viewport_lib::resources::{
    HDR_COLOR_FORMAT, MASK_COLOR_FORMAT, PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT,
    SCENE_DEPTH_FORMAT,
};
use wgpu::util::DeviceExt;

use crate::item::TerrainCollection;
use crate::mesh;
use crate::pick;

/// Per-layer uniform laid out to match `TerrainLayer` in `terrain.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct LayerUniform {
    albedo: [f32; 3],
    metallic: f32,
    roughness: f32,
    height_bias: f32,
    textured: f32,
    normal_mapped: f32,
    uv_scale: [f32; 2],
    uv_offset: [f32; 2],
    normal_scale: f32,
    mask_mapped: f32,
    _pad0: f32,
    _pad1: f32,
    mask_remap_min: [f32; 4],
    mask_remap_max: [f32; 4],
}

/// Per-object uniform matching `TerrainObject` in `terrain.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    layers: [LayerUniform; 8],
    height_blend_strength: f32,
    height_blend_noise_scale: f32,
    /// Item pick id, read by the `vs_pick` stage so terrain draws into the GPU
    /// pick pass.
    pick_id: u32,
    _pad1: f32,
}

/// One LOD level of one patch.
struct PatchLod {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
}

/// One patch of a terrain: a fixed-size grid square baked at every
/// LOD level. The renderer picks one LOD per frame based on camera
/// distance.
struct BakedPatch {
    lods: Vec<PatchLod>,
    aabb_min: glam::Vec3,
    aabb_max: glam::Vec3,
    /// Picked at cull() time; consumed by paint() / outline_mask /
    /// shadow.
    chosen_lod: u32,
    /// Set by cull() when the patch is outside the frustum.
    visible: bool,
}

/// Baked GPU state for one terrain.
struct BakedTerrain {
    uniform_buf: wgpu::Buffer,
    object_bg: wgpu::BindGroup,
    splatmap_a_tex: wgpu::Texture,
    splatmap_b_tex: wgpu::Texture,
    layer_tex: wgpu::Texture,
    normal_tex: wgpu::Texture,
    mask_tex: wgpu::Texture,
    patches: Vec<BakedPatch>,
    /// Patch grid dimensions in patches. Retained for diagnostics
    /// and future per-patch tooling.
    #[allow(dead_code)]
    patch_grid: [u32; 2],
    cpu: CpuHeightmap,
    sig: BakeSig,
    splat_sig: SplatSig,
    tex_sig: LayerTexSig,
    norm_sig: LayerTexSig,
    mask_sig: LayerTexSig,
}

#[derive(Clone, PartialEq)]
struct BakeSig {
    dims: [u32; 2],
    world_size: [f32; 2],
    height_range: [f32; 2],
    origin: [f32; 3],
    heightmap_version: u64,
    /// LOD policy bits that participate in the bake. Changing any of
    /// these triggers a full patch + LOD rebake.
    patch_cells: u32,
    max_lod: u32,
    skirt_depth_q: i32,
}

#[derive(Clone, PartialEq)]
struct SplatSig {
    dims_a: [u32; 2],
    dims_b: [u32; 2],
    version_a: u64,
    version_b: u64,
}

/// Signature of the diffuse layer array. `present` is false when the
/// item has no textures and a 1x1 placeholder array is bound instead.
#[derive(Clone, PartialEq)]
struct LayerTexSig {
    present: bool,
    dims: [u32; 2],
    version: u64,
}

impl LayerTexSig {
    fn of(textures: Option<&crate::item::LayerTextures>) -> Self {
        match textures {
            Some(t) => Self {
                present: true,
                dims: t.dims,
                version: t.version,
            },
            None => Self {
                present: false,
                dims: [1, 1],
                version: 0,
            },
        }
    }
}

/// Heightmap retained for CPU picking.
pub(crate) struct CpuHeightmap {
    pub heights: Vec<f32>,
    pub dims: [u32; 2],
    pub origin: glam::Vec3,
    pub world_size: [f32; 2],
}

pub struct TerrainPlugin {
    object_bgl: Option<wgpu::BindGroupLayout>,
    opaque_pipeline: Option<wgpu::RenderPipeline>,
    mask_pipeline: Option<wgpu::RenderPipeline>,
    pick_pipeline: Option<wgpu::RenderPipeline>,
    splatmap_sampler: Option<wgpu::Sampler>,
    albedo_sampler: Option<wgpu::Sampler>,
    baked: HashMap<PickId, BakedTerrain>,
    sample_count: u32,
    /// Patch size in heightmap cells per side at LOD 0. Default 32.
    patch_cells: u32,
    /// Number of LOD levels generated per patch. 1 disables LOD; the
    /// default of 4 produces decimation factors 1x / 2x / 4x / 8x.
    max_lod: u32,
    /// Skirt height in world units, dropped below each patch's
    /// perimeter to hide cracks between LOD neighbours. 0 disables
    /// skirts.
    skirt_depth: f32,
    /// World-space distance per LOD step. The patch picks LOD
    /// `floor(distance / lod_distance)` clamped to `0..max_lod`.
    lod_distance: f32,
}

impl Default for TerrainPlugin {
    fn default() -> Self {
        Self {
            object_bgl: None,
            opaque_pipeline: None,
            mask_pipeline: None,
            pick_pipeline: None,
            splatmap_sampler: None,
            albedo_sampler: None,
            baked: HashMap::new(),
            sample_count: 1,
            patch_cells: 32,
            max_lod: 4,
            skirt_depth: 1.0,
            lod_distance: 64.0,
        }
    }
}

impl TerrainPlugin {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sample_count(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }

    /// Patch size in heightmap cells per side at LOD 0. Default 32.
    /// Heightmap dimensions need not be a multiple; the final row /
    /// column of patches is clipped to fit.
    pub fn with_patch_cells(mut self, cells: u32) -> Self {
        self.patch_cells = cells.max(2);
        self
    }

    /// Number of LOD levels per patch. `1` disables LOD; the default
    /// 4 produces decimation factors 1x / 2x / 4x / 8x. Capped at 9.
    pub fn with_max_lod(mut self, max_lod: u32) -> Self {
        self.max_lod = max_lod.clamp(1, 9);
        self
    }

    /// Skirt height in world units. `0.0` disables skirts.
    pub fn with_skirt_depth(mut self, depth: f32) -> Self {
        self.skirt_depth = depth.max(0.0);
        self
    }

    /// World-space distance per LOD step. Smaller values push patches
    /// to lower LOD sooner (more aggressive decimation).
    pub fn with_lod_distance(mut self, distance: f32) -> Self {
        self.lod_distance = distance.max(1.0);
        self
    }
}

fn build_shader_source() -> String {
    let mut s = String::with_capacity(8 * 1024);
    s.push_str(shared_wgsl::SHARED_BINDINGS_WGSL);
    s.push('\n');
    s.push_str(shared_wgsl::SHARED_PBR_WGSL);
    s.push('\n');
    s.push_str(shared_wgsl::SHARED_MASK_WGSL);
    s.push('\n');
    s.push_str(shared_wgsl::SHARED_PICK_WGSL);
    s.push('\n');
    s.push_str(include_str!("terrain.wgsl"));
    s
}

impl ItemTypePlugin for TerrainPlugin {
    fn type_name(&self) -> &'static str {
        crate::TYPE_NAME
    }

    fn init_gpu(&mut self, device: &wgpu::Device, shared: &SharedBindings<'_>) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: wgpu::ShaderSource::Wgsl(build_shader_source().into()),
        });

        let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_object_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pipeline_layout"),
            bind_group_layouts: &[shared.group0_layout, &object_bgl],
            push_constant_ranges: &[],
        });

        let vbuf_layout = mesh::vertex_layout();

        let opaque_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_opaque_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&vbuf_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SCENE_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let mask_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_mask_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_mask"),
                buffers: std::slice::from_ref(&vbuf_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("viewport_mask_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SCENE_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        // Pick pipeline: matches the lib's three-target pick pass (object id,
        // primitive id, depth) plus the scene depth-stencil. vs_pick forwards
        // the object pick id; viewport_pick_fs (from SHARED_PICK_WGSL) writes it.
        let pick_color = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let pick_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_pick_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_pick"),
                buffers: std::slice::from_ref(&vbuf_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("viewport_pick_fs"),
                targets: &[
                    pick_color(PICK_COLOR_FORMAT),
                    pick_color(PICK_COLOR_FORMAT),
                    pick_color(PICK_DEPTH_CHANNEL_FORMAT),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SCENE_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let splatmap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_splatmap_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Diffuse, normal, and mask layers tile across the surface, so
        // this sampler wraps and filters trilinearly. Anisotropy keeps
        // the tiling sharp at grazing angles instead of smearing.
        let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_albedo_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });

        self.object_bgl = Some(object_bgl);
        self.opaque_pipeline = Some(opaque_pipeline);
        self.mask_pipeline = Some(mask_pipeline);
        self.pick_pipeline = Some(pick_pipeline);
        self.splatmap_sampler = Some(splatmap_sampler);
        self.albedo_sampler = Some(albedo_sampler);
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(coll) = items.as_any().downcast_ref::<TerrainCollection>() else {
            return Vec::new();
        };
        let Some(object_bgl) = self.object_bgl.as_ref() else {
            return Vec::new();
        };
        let Some(sampler) = self.splatmap_sampler.as_ref() else {
            return Vec::new();
        };
        let Some(albedo_sampler) = self.albedo_sampler.as_ref() else {
            return Vec::new();
        };
        let mut seen: Vec<PickId> = Vec::with_capacity(coll.items.len());
        for item in &coll.items {
            let key = item.settings.pick_id;
            seen.push(key);

            let sig = BakeSig {
                dims: item.dims,
                world_size: item.world_size,
                height_range: item.height_range,
                origin: item.origin.to_array(),
                heightmap_version: item.heightmap_version,
                patch_cells: self.patch_cells,
                max_lod: self.max_lod,
                // Quantise the skirt depth so float wobble does not
                // trigger spurious rebakes.
                skirt_depth_q: (self.skirt_depth * 1024.0) as i32,
            };
            let splat_sig = SplatSig {
                dims_a: item.splatmaps[0].dims,
                dims_b: item.splatmaps[1].dims,
                version_a: item.splatmaps[0].version,
                version_b: item.splatmaps[1].version,
            };
            let tex_sig = LayerTexSig::of(item.layer_textures.as_ref());
            let norm_sig = LayerTexSig::of(item.normal_textures.as_ref());
            let mask_sig = LayerTexSig::of(item.mask_textures.as_ref());

            let needs_mesh_rebake = self.baked.get(&key).is_none_or(|b| b.sig != sig);
            let needs_splat_reupload = self
                .baked
                .get(&key)
                .is_none_or(|b| b.splat_sig != splat_sig);
            let needs_tex_reupload = self.baked.get(&key).is_none_or(|b| b.tex_sig != tex_sig);
            let needs_norm_reupload = self.baked.get(&key).is_none_or(|b| b.norm_sig != norm_sig);
            let needs_mask_reupload = self.baked.get(&key).is_none_or(|b| b.mask_sig != mask_sig);

            if needs_mesh_rebake {
                let (patches, patch_grid) = bake_patches(
                    device,
                    item,
                    self.patch_cells,
                    self.max_lod,
                    self.skirt_depth,
                );
                let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("terrain_object_uniform"),
                    size: std::mem::size_of::<ObjectUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let splatmap_a_tex = create_splatmap_texture(device, &item.splatmaps[0].dims);
                let splatmap_b_tex = create_splatmap_texture(device, &item.splatmaps[1].dims);
                upload_splatmap(
                    queue,
                    &splatmap_a_tex,
                    item.splatmaps[0].rgba(),
                    &item.splatmaps[0].dims,
                );
                upload_splatmap(
                    queue,
                    &splatmap_b_tex,
                    item.splatmaps[1].rgba(),
                    &item.splatmaps[1].dims,
                );
                let layer_tex =
                    create_layer_array_texture(device, &tex_sig.dims, LayerArrayKind::Albedo);
                upload_layer_array(
                    queue,
                    &layer_tex,
                    item.layer_textures.as_ref(),
                    &tex_sig.dims,
                );
                let normal_tex =
                    create_layer_array_texture(device, &norm_sig.dims, LayerArrayKind::Normal);
                upload_layer_array(
                    queue,
                    &normal_tex,
                    item.normal_textures.as_ref(),
                    &norm_sig.dims,
                );
                let mask_tex =
                    create_layer_array_texture(device, &mask_sig.dims, LayerArrayKind::Mask);
                upload_layer_array(
                    queue,
                    &mask_tex,
                    item.mask_textures.as_ref(),
                    &mask_sig.dims,
                );
                let object_bg = build_object_bind_group(
                    device,
                    object_bgl,
                    &uniform_buf,
                    &splatmap_a_tex,
                    &splatmap_b_tex,
                    sampler,
                    &layer_tex,
                    albedo_sampler,
                    &normal_tex,
                    &mask_tex,
                );
                let heights: Vec<f32> = decode_heights(item);
                self.baked.insert(
                    key,
                    BakedTerrain {
                        uniform_buf,
                        object_bg,
                        splatmap_a_tex,
                        splatmap_b_tex,
                        layer_tex,
                        normal_tex,
                        mask_tex,
                        patches,
                        patch_grid,
                        cpu: CpuHeightmap {
                            heights,
                            dims: item.dims,
                            origin: item.origin,
                            world_size: item.world_size,
                        },
                        sig: sig.clone(),
                        splat_sig: splat_sig.clone(),
                        tex_sig: tex_sig.clone(),
                        norm_sig: norm_sig.clone(),
                        mask_sig: mask_sig.clone(),
                    },
                );
            } else if needs_splat_reupload
                || needs_tex_reupload
                || needs_norm_reupload
                || needs_mask_reupload
            {
                let entry = self.baked.get_mut(&key).unwrap();
                let mut rebind = false;

                if needs_splat_reupload {
                    let a_dims_changed = entry.splat_sig.dims_a != splat_sig.dims_a;
                    let b_dims_changed = entry.splat_sig.dims_b != splat_sig.dims_b;
                    if a_dims_changed {
                        entry.splatmap_a_tex =
                            create_splatmap_texture(device, &item.splatmaps[0].dims);
                    }
                    if b_dims_changed {
                        entry.splatmap_b_tex =
                            create_splatmap_texture(device, &item.splatmaps[1].dims);
                    }
                    upload_splatmap(
                        queue,
                        &entry.splatmap_a_tex,
                        item.splatmaps[0].rgba(),
                        &item.splatmaps[0].dims,
                    );
                    upload_splatmap(
                        queue,
                        &entry.splatmap_b_tex,
                        item.splatmaps[1].rgba(),
                        &item.splatmaps[1].dims,
                    );
                    rebind |= a_dims_changed || b_dims_changed;
                    entry.splat_sig = splat_sig;
                }

                if needs_tex_reupload {
                    if entry.tex_sig.dims != tex_sig.dims {
                        entry.layer_tex = create_layer_array_texture(
                            device,
                            &tex_sig.dims,
                            LayerArrayKind::Albedo,
                        );
                        rebind = true;
                    }
                    upload_layer_array(
                        queue,
                        &entry.layer_tex,
                        item.layer_textures.as_ref(),
                        &tex_sig.dims,
                    );
                    entry.tex_sig = tex_sig;
                }

                if needs_norm_reupload {
                    if entry.norm_sig.dims != norm_sig.dims {
                        entry.normal_tex = create_layer_array_texture(
                            device,
                            &norm_sig.dims,
                            LayerArrayKind::Normal,
                        );
                        rebind = true;
                    }
                    upload_layer_array(
                        queue,
                        &entry.normal_tex,
                        item.normal_textures.as_ref(),
                        &norm_sig.dims,
                    );
                    entry.norm_sig = norm_sig;
                }

                if needs_mask_reupload {
                    if entry.mask_sig.dims != mask_sig.dims {
                        entry.mask_tex = create_layer_array_texture(
                            device,
                            &mask_sig.dims,
                            LayerArrayKind::Mask,
                        );
                        rebind = true;
                    }
                    upload_layer_array(
                        queue,
                        &entry.mask_tex,
                        item.mask_textures.as_ref(),
                        &mask_sig.dims,
                    );
                    entry.mask_sig = mask_sig;
                }

                if rebind {
                    entry.object_bg = build_object_bind_group(
                        device,
                        object_bgl,
                        &entry.uniform_buf,
                        &entry.splatmap_a_tex,
                        &entry.splatmap_b_tex,
                        sampler,
                        &entry.layer_tex,
                        albedo_sampler,
                        &entry.normal_tex,
                        &entry.mask_tex,
                    );
                }
            }

            let baked = self.baked.get(&key).unwrap();
            let uniform = ObjectUniform {
                model: glam::Mat4::IDENTITY.to_cols_array_2d(),
                layers: std::array::from_fn(|i| layer_uniform(&item.surface_layers[i])),
                height_blend_strength: item.height_blend_strength,
                height_blend_noise_scale: item.height_blend_noise_scale,
                pick_id: key.0 as u32,
                _pad1: 0.0,
            };
            queue.write_buffer(&baked.uniform_buf, 0, bytemuck::bytes_of(&uniform));
        }

        self.baked.retain(|k, _| seen.contains(k));

        Vec::new()
    }

    fn cull(
        &mut self,
        frustum: &viewport_lib::camera::frustum::Frustum,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        let Some(coll) = items.as_any().downcast_ref::<TerrainCollection>() else {
            return;
        };
        let lod_distance = self.lod_distance.max(1.0);
        let max_lod = self.max_lod.saturating_sub(1);
        let eye = glam::Vec3::from(ctx.camera.eye_position);
        for item in &coll.items {
            let key = item.settings.pick_id;
            let Some(baked) = self.baked.get_mut(&key) else {
                continue;
            };
            for patch in baked.patches.iter_mut() {
                let aabb = viewport_lib::Aabb {
                    min: patch.aabb_min,
                    max: patch.aabb_max,
                };
                patch.visible = !frustum.cull_aabb(&aabb);
                if patch.visible {
                    let centre = (patch.aabb_min + patch.aabb_max) * 0.5;
                    let d = centre.distance(eye);
                    let lod = (d / lod_distance) as u32;
                    patch.chosen_lod = lod.min(max_lod);
                }
            }
        }
    }

    fn paint<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        _ctx: &PaintContext<'a>,
        items: &'a dyn PluginItemCollection,
    ) {
        let Some(coll) = items.as_any().downcast_ref::<TerrainCollection>() else {
            return;
        };
        let Some(pipeline) = self.opaque_pipeline.as_ref() else {
            return;
        };
        pass.set_pipeline(pipeline);
        for item in &coll.items {
            if item.settings.hidden {
                continue;
            }
            let key = item.settings.pick_id;
            let Some(baked) = self.baked.get(&key) else {
                continue;
            };
            pass.set_bind_group(1, &baked.object_bg, &[]);
            for patch in &baked.patches {
                if !patch.visible {
                    continue;
                }
                let lod = (patch.chosen_lod as usize).min(patch.lods.len() - 1);
                let mesh = &patch.lods[lod];
                pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }

    fn outline_mask<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        _ctx: &OutlineMaskContext<'a>,
        items: &'a dyn PluginItemCollection,
    ) {
        let Some(coll) = items.as_any().downcast_ref::<TerrainCollection>() else {
            return;
        };
        let Some(pipeline) = self.mask_pipeline.as_ref() else {
            return;
        };
        let mut bound = false;
        for item in &coll.items {
            if item.settings.hidden || !item.settings.selected {
                continue;
            }
            let key = item.settings.pick_id;
            let Some(baked) = self.baked.get(&key) else {
                continue;
            };
            if !bound {
                pass.set_pipeline(pipeline);
                bound = true;
            }
            pass.set_bind_group(1, &baked.object_bg, &[]);
            for patch in &baked.patches {
                if !patch.visible {
                    continue;
                }
                let lod = (patch.chosen_lod as usize).min(patch.lods.len() - 1);
                let mesh = &patch.lods[lod];
                pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }

    fn render_pick<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        _ctx: &PickPassContext<'a>,
        items: &'a dyn PluginItemCollection,
    ) {
        let Some(coll) = items.as_any().downcast_ref::<TerrainCollection>() else {
            return;
        };
        let Some(pipeline) = self.pick_pipeline.as_ref() else {
            return;
        };
        let mut bound = false;
        for item in &coll.items {
            if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                continue;
            }
            let key = item.settings.pick_id;
            let Some(baked) = self.baked.get(&key) else {
                continue;
            };
            if !bound {
                pass.set_pipeline(pipeline);
                bound = true;
            }
            // Group 0 (camera) is bound by the lib; group 1 carries the object
            // uniform, whose pick_id vs_pick forwards.
            pass.set_bind_group(1, &baked.object_bg, &[]);
            for patch in &baked.patches {
                if !patch.visible {
                    continue;
                }
                let lod = (patch.chosen_lod as usize).min(patch.lods.len() - 1);
                let mesh = &patch.lods[lod];
                pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }

    fn pick(&self, ray: &PickRay) -> Option<(f32, viewport_lib::PickHit)> {
        let mut best: Option<(f32, viewport_lib::PickHit)> = None;
        for (pick_id, baked) in self.baked.iter() {
            if let Some((t, hit)) = pick::pick_heightmap(ray, &baked.cpu, *pick_id) {
                if best.as_ref().is_none_or(|(bt, _)| t < *bt) {
                    best = Some((t, hit));
                }
            }
        }
        best
    }
}

fn layer_uniform(layer: &crate::item::TerrainLayer) -> LayerUniform {
    LayerUniform {
        albedo: layer.albedo,
        metallic: layer.metallic,
        roughness: layer.roughness,
        height_bias: layer.height_bias,
        textured: if layer.textured { 1.0 } else { 0.0 },
        normal_mapped: if layer.normal_mapped { 1.0 } else { 0.0 },
        uv_scale: layer.uv_scale,
        uv_offset: layer.uv_offset,
        normal_scale: layer.normal_scale,
        mask_mapped: if layer.mask_mapped { 1.0 } else { 0.0 },
        _pad0: 0.0,
        _pad1: 0.0,
        mask_remap_min: layer.mask_remap_min,
        mask_remap_max: layer.mask_remap_max,
    }
}

/// Generate a grid of patches and their LOD meshes from a heightmap.
/// Returns `(patches, [patches_x, patches_y])`.
fn bake_patches(
    device: &wgpu::Device,
    item: &crate::item::TerrainItem,
    patch_cells: u32,
    max_lod: u32,
    skirt_depth: f32,
) -> (Vec<BakedPatch>, [u32; 2]) {
    let cells_w = item.dims[0].saturating_sub(1);
    let cells_h = item.dims[1].saturating_sub(1);
    let px = cells_w.div_ceil(patch_cells);
    let py = cells_h.div_ceil(patch_cells);

    let mut patches = Vec::with_capacity((px * py) as usize);
    for ipy in 0..py {
        for ipx in 0..px {
            let origin = [ipx * patch_cells, ipy * patch_cells];
            let mut lods = Vec::with_capacity(max_lod as usize);
            let mut patch_aabb_min = glam::Vec3::splat(f32::INFINITY);
            let mut patch_aabb_max = glam::Vec3::splat(f32::NEG_INFINITY);
            for lod in 0..max_lod {
                let mesh = crate::mesh::build_patch(item, origin, patch_cells, lod, skirt_depth);
                patch_aabb_min = patch_aabb_min.min(mesh.aabb_min);
                patch_aabb_max = patch_aabb_max.max(mesh.aabb_max);
                let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("terrain_patch_vbuf"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("terrain_patch_ibuf"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                lods.push(PatchLod {
                    vbuf,
                    ibuf,
                    index_count: mesh.indices.len() as u32,
                });
            }
            patches.push(BakedPatch {
                lods,
                aabb_min: patch_aabb_min,
                aabb_max: patch_aabb_max,
                chosen_lod: 0,
                visible: true,
            });
        }
    }
    (patches, [px, py])
}

#[allow(clippy::too_many_arguments)]
fn build_object_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    splatmap_a: &wgpu::Texture,
    splatmap_b: &wgpu::Texture,
    sampler: &wgpu::Sampler,
    layer_tex: &wgpu::Texture,
    albedo_sampler: &wgpu::Sampler,
    normal_tex: &wgpu::Texture,
    mask_tex: &wgpu::Texture,
) -> wgpu::BindGroup {
    let view_a = splatmap_a.create_view(&wgpu::TextureViewDescriptor::default());
    let view_b = splatmap_b.create_view(&wgpu::TextureViewDescriptor::default());
    let array_view = |tex: &wgpu::Texture| {
        tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    };
    let view_layers = array_view(layer_tex);
    let view_normals = array_view(normal_tex);
    let view_masks = array_view(mask_tex);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain_object_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view_a),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&view_b),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&view_layers),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(albedo_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&view_normals),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&view_masks),
            },
        ],
    })
}

fn decode_heights(item: &crate::item::TerrainItem) -> Vec<f32> {
    let (lo, hi) = (item.height_range[0], item.height_range[1]);
    let span = hi - lo;
    item.heightmap
        .iter()
        .map(|&s| item.origin.z + lo + (s as f32 / u16::MAX as f32) * span)
        .collect()
}

fn create_splatmap_texture(device: &wgpu::Device, dims: &[u32; 2]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain_splatmap"),
        size: wgpu::Extent3d {
            width: dims[0].max(1),
            height: dims[1].max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Which per-layer array is being built. Diffuse holds sRGB colour;
/// normal holds linear tangent-space vectors.
#[derive(Copy, Clone)]
enum LayerArrayKind {
    Albedo,
    Normal,
    Mask,
}

/// Number of mip levels for a texture of the given size: a full chain
/// down to 1x1. Without this the tiling detail textures alias into
/// salt-and-pepper noise at a distance.
fn mip_count(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

/// Create a per-layer texture array: one RGBA slice per surface layer at
/// `dims`, with a full mip chain. Diffuse arrays are sRGB, normal and
/// mask arrays are linear.
fn create_layer_array_texture(
    device: &wgpu::Device,
    dims: &[u32; 2],
    kind: LayerArrayKind,
) -> wgpu::Texture {
    let (label, format) = match kind {
        LayerArrayKind::Albedo => ("terrain_layer_array", wgpu::TextureFormat::Rgba8UnormSrgb),
        LayerArrayKind::Normal => ("terrain_normal_array", wgpu::TextureFormat::Rgba8Unorm),
        LayerArrayKind::Mask => ("terrain_mask_array", wgpu::TextureFormat::Rgba8Unorm),
    };
    let w = dims[0].max(1);
    let h = dims[1].max(1);
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: crate::item::LAYER_COUNT as u32,
        },
        mip_level_count: mip_count(w, h),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Box-downsample an RGBA image to half size (rounding down, minimum 1).
fn downsample_half(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let dw = (w / 2).max(1);
    let dh = (h / 2).max(1);
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    for y in 0..dh as usize {
        for x in 0..dw as usize {
            let sx0 = (x * 2).min(w - 1);
            let sy0 = (y * 2).min(h - 1);
            let sx1 = (sx0 + 1).min(w - 1);
            let sy1 = (sy0 + 1).min(h - 1);
            let o = (y * dw as usize + x) * 4;
            for c in 0..4 {
                let sum = src[(sy0 * w + sx0) * 4 + c] as u32
                    + src[(sy0 * w + sx1) * 4 + c] as u32
                    + src[(sy1 * w + sx0) * 4 + c] as u32
                    + src[(sy1 * w + sx1) * 4 + c] as u32;
                out[o + c] = (sum / 4) as u8;
            }
        }
    }
    (out, dw, dh)
}

/// Upload the packed layer array, generating a mip chain per layer so
/// the tiling textures filter cleanly at a distance. With no textures
/// present the array is a 1x1 placeholder left at its cleared (zero)
/// contents; every layer falls back to its flat value in the shader, so
/// nothing needs writing.
fn upload_layer_array(
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    textures: Option<&crate::item::LayerTextures>,
    dims: &[u32; 2],
) {
    let Some(textures) = textures else {
        return;
    };
    let w = dims[0].max(1);
    let h = dims[1].max(1);
    let slot_len = (w as usize) * (h as usize) * 4;
    let rgba = textures.rgba();
    if rgba.len() != slot_len * crate::item::LAYER_COUNT {
        return;
    }
    let levels = mip_count(w, h);
    for layer in 0..crate::item::LAYER_COUNT {
        // Level 0 is the source slice; each further level halves the
        // previous one.
        let mut level_data = rgba[layer * slot_len..(layer + 1) * slot_len].to_vec();
        let mut lw = w;
        let mut lh = h;
        for level in 0..levels {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &level_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(lw * 4),
                    rows_per_image: Some(lh),
                },
                wgpu::Extent3d {
                    width: lw,
                    height: lh,
                    depth_or_array_layers: 1,
                },
            );
            if level + 1 < levels {
                let (next, nw, nh) = downsample_half(&level_data, lw, lh);
                level_data = next;
                lw = nw;
                lh = nh;
            }
        }
    }
}

fn upload_splatmap(queue: &wgpu::Queue, tex: &wgpu::Texture, rgba: &[u8], dims: &[u32; 2]) {
    let w = dims[0].max(1);
    let h = dims[1].max(1);
    let expected = (w * h * 4) as usize;
    let mut staging: Vec<u8>;
    let slice: &[u8] = if rgba.len() == expected {
        rgba
    } else {
        staging = vec![0u8; expected];
        let copy = rgba.len().min(expected);
        staging[..copy].copy_from_slice(&rgba[..copy]);
        &staging
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        slice,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_source_validates() {
        // The concatenated terrain shader is only compiled when a device
        // is created, which the unit tests do not do. Validate it here so
        // WGSL mistakes (including non-uniform texture sampling in the
        // layer blend loop) fail the test suite rather than at runtime.
        let src = build_shader_source();
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("terrain shader failed to parse: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("terrain shader failed to validate: {e}"));
    }

    #[test]
    fn mip_count_covers_full_chain() {
        assert_eq!(mip_count(1, 1), 1);
        assert_eq!(mip_count(1024, 1024), 11);
        assert_eq!(mip_count(1024, 512), 11);
        assert_eq!(mip_count(512, 512), 10);
    }

    #[test]
    fn downsample_half_averages_and_halves() {
        // 2x2 image, one white pixel and three black: the single
        // downsampled pixel is the average (255/4 rounded down = 63).
        let src = vec![255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let (out, w, h) = downsample_half(&src, 2, 2);
        assert_eq!((w, h), (1, 1));
        assert_eq!(out, vec![63, 63, 63, 63]);
    }

    #[test]
    fn layer_uniform_matches_wgsl_size() {
        // TerrainLayer in terrain.wgsl rounds to 96 bytes under std140.
        // The Rust mirror must match or the per-layer array misaligns on
        // the GPU.
        assert_eq!(std::mem::size_of::<LayerUniform>(), 96);
    }
}
