struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
@group(2) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) tangent_position: vec3<f32>,
    @location(2) tangent_light_position: vec3<f32>,
    @location(3) tangent_view_position: vec3<f32>,
};

@vertex
fn vs_main(
    v_input: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {

    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    // Construct the tangent matrix
    let world_normal = normalize(normal_matrix * v_input.normal);
    let world_tangent = normalize(normal_matrix * v_input.tangent);
    let world_bitangent = normalize(normal_matrix * v_input.bitangent);
    let tangent_matrix = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal,
    ));

    let world_position = model_matrix * vec4<f32>(v_input.position, 1.0);

    var v_out: VertexOutput;
    v_out.clip_position = camera.view_proj * world_position;
    v_out.tex_coords = v_input.tex_coords;
    
    v_out.tangent_position = tangent_matrix * world_position.xyz;
    v_out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;
    v_out.tangent_light_position = tangent_matrix * light.position;
    return v_out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> 
{
    let center_pos: vec2<f32> = in.tex_coords - 0.5;
    let distance: f32 = length(center_pos);

    let inverse_distance = (1 - 2 * distance);
    let alpha_mask = step(0.0, inverse_distance);

    let color = vec3<f32>(0.7, 0.7, 0.7);

    return vec4<f32>(result, alpha_mask);
}