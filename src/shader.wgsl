
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    v_input: VertexInput,
) -> VertexOutput {
    var v_out: VertexOutput;
    v_out.tex_coords = v_input.tex_coords;
    v_out.clip_position = vec4<f32>(v_input.position, 1.0);
    return v_out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> 
{
    // Note!!!!!!!!!!!!!!!!!
    // I had to multiply by (1, -1) because I'm too lazy to do this in CPU code, lmao
    return textureSample(t_diffuse, s_diffuse, in.tex_coords * vec2<f32>(1.,-1.));
}
 
/* 
Something to note about @builtin(position), 
in the fragment shader this value is in framebuffer space (opens new window). 

This means that if your window is 800x600, the x and y of clip_position would be between 
0-800 and 0-600 respectively with the y = 0 being the top of the screen. 

This can be useful if you want to know pixel coordinates of a given fragment, 
but if you want the position coordinates you'll have to pass them in separately.
*/