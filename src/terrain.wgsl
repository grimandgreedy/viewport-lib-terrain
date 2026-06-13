// Terrain vertex + fragment entries. Concatenated with viewport-lib's
// SHARED_BINDINGS_WGSL, SHARED_PBR_WGSL, and SHARED_MASK_WGSL at runtime.

struct TerrainLayer {
    albedo:      vec3<f32>,
    metallic:    f32,
    roughness:   f32,
    height_bias: f32,
    _pad0:       f32,
    _pad1:       f32,
};

struct TerrainObject {
    model:                     mat4x4<f32>,
    layers:                    array<TerrainLayer, 8>,
    height_blend_strength:     f32,
    height_blend_noise_scale:  f32,
    _pad0:                     f32,
    _pad1:                     f32,
};

@group(1) @binding(0) var<uniform> obj:            TerrainObject;
@group(1) @binding(1) var          splatmap_a_tex: texture_2d<f32>;
@group(1) @binding(2) var          splatmap_b_tex: texture_2d<f32>;
@group(1) @binding(3) var          splatmap_samp:  sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       uv:           vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = obj.model * vec4<f32>(in.position, 1.0);
    let n_world = normalize((obj.model * vec4<f32>(in.normal, 0.0)).xyz);
    var out: VsOut;
    out.clip_pos     = camera.view_proj * world;
    out.world_pos    = world.xyz;
    out.world_normal = n_world;
    out.uv           = in.uv;
    return out;
}

// Cheap 2D value noise used to inject per-pixel height variation so the
// height-blend produces irregular, pebble-edged transitions instead of
// straight lines along splatmap boundaries.
fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q = q + dot(q, q + 45.32);
    return fract(q.x * q.y);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if !viewport_clip_test(in.world_pos) { discard; }

    let view_dir = normalize(camera.eye_pos - in.world_pos);
    let n_raw    = normalize(in.world_normal);
    let n_lit    = select(n_raw, -n_raw, dot(n_raw, view_dir) < 0.0);

    // Sample both splatmaps into eight per-layer weights.
    let s0 = textureSample(splatmap_a_tex, splatmap_samp, in.uv);
    let s1 = textureSample(splatmap_b_tex, splatmap_samp, in.uv);
    var w: array<f32, 8> = array<f32, 8>(
        s0.r, s0.g, s0.b, s0.a,
        s1.r, s1.g, s1.b, s1.a,
    );

    // Normalise to a unit sum. Fall back to layer 0 if everything is
    // zero so unpainted pixels still shade.
    var total = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        total = total + w[i];
    }
    if total < 1e-5 {
        w[0] = 1.0;
        total = 1.0;
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        w[i] = w[i] / total;
    }

    // Optional height-blend: softmax the per-layer (height_bias + noise)
    // with `height_blend_strength` as temperature. Larger strength
    // sharpens the winning layer's region.
    if obj.height_blend_strength > 0.0 {
        let n = value_noise(in.uv * obj.height_blend_noise_scale);
        var max_h: f32 = -1e9;
        for (var i = 0u; i < 8u; i = i + 1u) {
            if w[i] > 0.0 {
                let h = obj.layers[i].height_bias + n;
                if h > max_h { max_h = h; }
            }
        }
        var sum_h = 0.0;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let h = obj.layers[i].height_bias + n;
            let v = w[i] * exp((h - max_h) * obj.height_blend_strength);
            w[i] = v;
            sum_h = sum_h + v;
        }
        let inv = 1.0 / max(sum_h, 1e-6);
        for (var i = 0u; i < 8u; i = i + 1u) {
            w[i] = w[i] * inv;
        }
    }

    var albedo    = vec3<f32>(0.0);
    var metallic  = 0.0;
    var roughness = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let l = obj.layers[i];
        albedo    = albedo    + l.albedo    * w[i];
        metallic  = metallic  + l.metallic  * w[i];
        roughness = roughness + l.roughness * w[i];
    }

    var inputs: PbrInputs;
    inputs.world_pos = in.world_pos;
    inputs.world_n   = n_lit;
    inputs.view_dir  = view_dir;
    inputs.albedo    = albedo;
    inputs.metallic  = metallic;
    inputs.roughness = roughness;
    inputs.ao        = 1.0;
    inputs.emissive  = vec3<f32>(0.0);

    let lit = viewport_pbr_shade(inputs);
    return vec4<f32>(lit, 1.0);
}

// Outline-mask vertex stage. Fragment uses viewport_mask_fs from
// SHARED_MASK_WGSL (constant 1.0 into the R8 mask).
@vertex
fn vs_mask(in: VsIn) -> @builtin(position) vec4<f32> {
    let world = obj.model * vec4<f32>(in.position, 1.0);
    return camera.view_proj * world;
}
