struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normals: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normals: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.normals = model.normals;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> 
{
    let center_pos: vec2<f32> = in.tex_coords - 0.5;
    let distance: f32 = length(center_pos);

    let inverse_distance: f32 = (1.0 - 2. * distance);
    let alpha_mask: f32 = step(0.0, inverse_distance);

    let color = vec3<f32>(0.7, 0.7, 0.7);

    return vec4<f32>(color, alpha_mask);
}