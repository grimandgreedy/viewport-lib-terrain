// Terrain vertex + fragment entries. Concatenated with viewport-lib's
// SHARED_BINDINGS_WGSL, SHARED_PBR_WGSL, and SHARED_MASK_WGSL at runtime.

struct TerrainLayer {
    albedo:        vec3<f32>,
    metallic:      f32,
    roughness:     f32,
    height_bias:   f32,
    textured:      f32,
    normal_mapped: f32,
    uv_scale:      vec2<f32>,
    uv_offset:     vec2<f32>,
    normal_scale:  f32,
    _pad0:         f32,
    _pad1:         f32,
    _pad2:         f32,
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
@group(1) @binding(4) var          albedo_array:   texture_2d_array<f32>;
@group(1) @binding(5) var          albedo_samp:    sampler;
@group(1) @binding(6) var          normal_array:   texture_2d_array<f32>;

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
    var ts_normal = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let l = obj.layers[i];
        // Sample every layer unconditionally so the texture fetch stays
        // in uniform control flow, then pick the textured or flat value
        // by the layer's flag. uv_scale is clamped so an untextured
        // layer (scale 0) cannot produce a non-finite coordinate.
        let scale     = max(l.uv_scale, vec2<f32>(1e-4, 1e-4));
        let layer_uv  = in.world_pos.xy / scale + l.uv_offset;
        let sampled   = textureSample(albedo_array, albedo_samp, layer_uv, i).rgb;
        let layer_col = mix(l.albedo, sampled, l.textured);
        albedo    = albedo    + layer_col   * w[i];
        metallic  = metallic  + l.metallic  * w[i];
        roughness = roughness + l.roughness * w[i];

        // Tangent-space normal: unpack to [-1, 1], scale the XY tilt, and
        // fall back to flat (0, 0, 1) when the layer has no normal map.
        // Accumulate the weighted result and normalise once after the
        // loop, which blends the painted layers without seams.
        let raw_n  = textureSample(normal_array, albedo_samp, layer_uv, i).xyz * 2.0 - 1.0;
        let tilted = vec3<f32>(raw_n.xy * l.normal_scale, max(raw_n.z, 1e-3));
        let n_ts   = mix(vec3<f32>(0.0, 0.0, 1.0), tilted, l.normal_mapped);
        ts_normal  = ts_normal + n_ts * w[i];
    }

    // Rebuild the world normal from the blended tangent-space normal. The
    // layer UVs run along world X and Y, so the tangent frame is the
    // world axes projected onto the surface. A heightfield normal always
    // has positive Z, so the world-X projection never degenerates.
    let geo_n    = normalize(in.world_normal);
    let tangent  = normalize(vec3<f32>(1.0, 0.0, 0.0) - geo_n * geo_n.x);
    let bitangent = cross(geo_n, tangent);
    let ts       = normalize(ts_normal);
    let mapped_n = normalize(tangent * ts.x + bitangent * ts.y + geo_n * ts.z);
    let shade_n  = select(mapped_n, -mapped_n, dot(mapped_n, view_dir) < 0.0);

    var inputs: PbrInputs;
    inputs.world_pos = in.world_pos;
    inputs.world_n   = shade_n;
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
