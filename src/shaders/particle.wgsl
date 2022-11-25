struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    aspect_ratio: f32,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct InstanceInput {
    // Model Matrix
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    // Normal Matrix
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
    @location(12) transparency: f32,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normals: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normals: vec3<f32>,
    @location(2) transparency: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {

    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    let world_position = model_matrix * vec4<f32>(model.position, 1.0);
    let final_pos = camera.view_proj * world_position;

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.normals = model.normals;
    out.clip_position = final_pos;
    out.transparency = instance.transparency;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_pos: vec2<f32> = in.tex_coords - 0.5;
    let distance: f32 = length(center_pos);

    let inverse_distance: f32 = (1.0 - 4. * distance);
    let alpha_mask: f32 = step(0.01, inverse_distance);

    if alpha_mask < 0.3 {
        discard;
    }

    let alpha = saturate(in.transparency * inverse_distance);

    let color = vec4<f32>(alpha_mask, alpha_mask, alpha_mask, alpha);

    return vec4<f32>(color);
}