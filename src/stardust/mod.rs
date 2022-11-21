use cgmath::{Vector3, Zero};
use typed_builder::TypedBuilder;

#[allow(dead_code)]
pub const QUADRATIC_CENTERED: fn(f32) -> f32 = |x: f32| -4. * (x) * (x - 1.);
pub const ONE_MINUS_T: fn(f32) -> f32 = |x: f32| 1. - x;
pub const ZERO: fn(f32) -> f32 = |_: f32| 0.;
#[allow(dead_code)]
pub const IDENTITY: fn(f32) -> f32 = |x: f32| x;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}

impl QuadVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub const QUAD_VERTS: &[QuadVertex] = &[
    QuadVertex {
        position: [-0.5, -0.5, 0.0],
        tex_coords: [0.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    QuadVertex {
        position: [0.5, -0.5, 0.0],
        tex_coords: [1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
    },
    QuadVertex {
        position: [-0.5, 0.5, 0.0],
        tex_coords: [0.0, 1.0],
        normal: [0.0, 0.0, -1.0],
    },
    QuadVertex {
        position: [0.5, 0.5, 0.0],
        tex_coords: [1.0, 1.0],
        normal: [0.0, 0.0, -1.0],
    },
];

pub const QUAD_INDICES: &[u32] = &[0, 1, 2, 1, 3, 2];

pub fn get_mesh(device: &wgpu::Device) -> crate::model::Mesh {
    let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTS),
            usage: wgpu::BufferUsages::VERTEX,
        },
    );

    let index_buffer = wgpu::util::DeviceExt::create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        },
    );

    crate::model::Mesh {
        name: "Particle_Mesh".to_owned(),
        vertex_buffer,
        index_buffer,
        num_elements: QUAD_INDICES.len() as u32,
        material: 0,
    }
}

type Colour3 = Vector3<f32>;
type Vec3 = Vector3<f32>;

trait AddAssign {
    fn add_assign(&mut self, other: Self);
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }
}

// pub struct ParticleBuilder {
//     position: Option<Vec3>,
//     velocity: Option<Vec3>,
//     size: Option<f32>,
//     total_lifetime: Option<f32>,
//     color: Option<Colour3>,
//     force_constant: Option<Vec3>,
//     force_curve: Option<[fn(f32) -> f32; 4]>,
//     transparency: Option<f32>,
//     transparency_curve: Option<fn(f32) -> f32>,
// }

// #[allow(dead_code)]
// impl ParticleBuilder {
//     pub fn position(&mut self, position: Vec3) -> &mut Self {
//         self.position = Some(position);
//         self
//     }
//     pub fn velocity(&mut self, velocity: Vec3) -> &mut Self {
//         self.velocity = Some(velocity);
//         self
//     }
//     pub fn size(&mut self, size: f32) -> &mut Self {
//         self.size = Some(size);
//         self
//     }
//     pub fn total_lifetime(&mut self, total_lifetime: f32) -> &mut Self {
//         self.total_lifetime = Some(total_lifetime);
//         self
//     }
//     pub fn color(&mut self, color: Colour3) -> &mut Self {
//         self.color = Some(color);
//         self
//     }
//     pub fn force_constant(&mut self, force_constant: Vec3) -> &mut Self {
//         self.force_constant = Some(force_constant);
//         self
//     }
//     pub fn force_curve(&mut self, force_curve: [fn(f32) -> f32; 4]) -> &mut Self {
//         self.force_curve = Some(force_curve);
//         self
//     }
//     pub fn transparency(&mut self, transparency: f32) -> &mut Self {
//         self.transparency = Some(transparency);
//         self
//     }
//     pub fn transparency_curve(&mut self, transparency_curve: fn(f32) -> f32) -> &mut Self {
//         self.transparency_curve = Some(transparency_curve);
//         self
//     }

//     pub fn build(&mut self) -> Particle {
//         Particle {
//             position: self.position.unwrap_or(Vec3::zero()),
//             velocity: self.velocity.unwrap_or(Vec3::zero()),
//             size: self.size.unwrap_or(1.0),
//             lifetime: self.total_lifetime.unwrap_or(1.0),
//             total_lifetime: self.total_lifetime.unwrap_or(1.0),
//             color: self.color.unwrap_or(Vec3::new(0.7, 0.7, 0.7)),
//             force_constant: self.force_constant.unwrap_or(Vec3::zero()),
//             force_curve: self
//                 .force_curve
//                 .unwrap_or([ZERO,ZERO,ZERO,ZERO,]),
//             transparency: self.transparency.unwrap_or(1.0),
//             transparency_curve: self.transparency_curve.unwrap_or(ONE_MINUS_T),
//         }
//     }
// }

#[derive(TypedBuilder)]
pub struct Particle {
    #[builder(default = Vector3::zero())]
    pub position: Vec3,
    #[builder(default = Vector3::zero())]
    pub velocity: Vec3,
    #[builder(default = Vector3::zero())]
    pub force_constant: Vec3,
    #[builder(default = [ZERO; 4])]
    pub force_curve: [fn(f32) -> f32; 4],
    #[builder(default)]
    pub size: f32,
    #[builder(default = Vector3::zero())]
    pub color: Colour3,
    #[builder(default = 1.0)]
    pub transparency: f32,
    #[builder(default = 1.0)]
    pub simulation_speed: f32,
    #[builder(default = ONE_MINUS_T)]
    pub transparency_curve: fn(f32) -> f32,
    #[builder(default = 10.)]
    pub total_lifetime: f32,
    #[builder(default = 10.)]
    pub lifetime: f32,
    // shape?
}

impl Particle {
    // pub fn new() -> ParticleBuilder {
    //     ParticleBuilder {
    //         position: None,
    //         velocity: None,
    //         size: None,
    //         total_lifetime: None,
    //         color: None,
    //         force_constant: None,
    //         force_curve: None,
    //         transparency: None,
    //         transparency_curve: None,
    //     }
    // }
    pub fn update(&mut self, dt: f32) {
        let dt = dt * self.simulation_speed;
        self.lifetime -= dt;
        self.update_curve_values();

        self.velocity += self.force_constant * dt;
        self.position += self.velocity * dt;
    }

    // https://github.dev/gfx-rs/wgpu/tree/master/wgpu/examples/boids

    fn update_curve_values(&mut self) {
        let normalized_time = (self.total_lifetime - self.lifetime) / self.total_lifetime;

        self.transparency = (self.transparency_curve)(normalized_time);
    }
}

// Stolen Code here:
/*

impl Draw<Transparent2d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Transparent2d,
    ) {
        let (effects_meta, effect_bind_groups, pipeline_cache, views, effects) =
            self.params.get(world);

        let view_uniform = views.get(view).unwrap();
        let effects_meta = effects_meta.into_inner();
        let effect_bind_groups = effect_bind_groups.into_inner();
        let effect_batch = effects.get(item.entity).unwrap();

        if let Some(pipeline) = pipeline_cache
            .into_inner()
            .get_render_pipeline(item.pipeline)
        {
            pass.set_render_pipeline(pipeline);

            // Vertex buffer containing the particle model to draw. Generally a quad.
            pass.set_vertex_buffer(0, effects_meta.vertices.buffer().unwrap().slice(..));

            // View properties (camera matrix, etc.)
            pass.set_bind_group(
                0,
                effects_meta.view_bind_group.as_ref().unwrap(),
                &[view_uniform.offset],
            );

            // Particles buffer
            let dispatch_indirect_offset = effects_meta
                .gpu_dispatch_indirect_aligned_size
                .unwrap()
                .get()
                * effect_batch.buffer_index;

            pass.set_bind_group(
                1,
                effect_bind_groups
                    .render_particle_buffers
                    .get(&effect_batch.buffer_index)
                    .unwrap(),
                &[dispatch_indirect_offset],
            );

            // Particle texture
            if effect_batch
                .layout_flags
                .contains(LayoutFlags::PARTICLE_TEXTURE)
            {
                let image_handle = Handle::weak(effect_batch.image_handle_id);
                if let Some(bind_group) = effect_bind_groups.images.get(&image_handle) {
                    pass.set_bind_group(2, bind_group, &[]);
                } else {
                    // Texture not ready; skip this drawing for now
                    trace!(
                            "Particle texture bind group not available for batch buf={} slice={:?}. Skipping draw call.",
                            effect_batch.buffer_index,
                            effect_batch.slice
                        );
                    return; //continue;
                }
            }

            let vertex_count = effects_meta.vertices.len() as u32;
            let particle_count = effect_batch.slice.end - effect_batch.slice.start;

            trace!(
                "Draw {} particles with {} vertices per particle for batch from buffer #{}.",
                particle_count,
                vertex_count,
                effect_batch.buffer_index
            );
            pass.draw(0..vertex_count, 0..particle_count);
        }
    }
}

*/

// // Must refactor this - curves are meant to be points, which you can convert into a [f32 -> f32] closure
// pub struct Curve {
//     pub point_list: Vec<CurvePoint>,
//     pub closure: fn(f32) -> f32,
// }

// pub struct CurvePoint {
//     pub key: f32,
//     pub value: f32,
// }

// impl Curve {
//     pub fn vec3_curve(
//         start_value: Vec3,
//         end_value: Vec3,
//         t_value: f32,
//         curve_closure: Box<dyn Fn(f32) -> f32>,
//     ) -> Vec3 {
//         let lerp_value = (curve_closure)(t_value);

//         (1. - lerp_value) * start_value + (lerp_value) * end_value
//     }

//     pub fn float_curve(
//         start_value: f32,
//         end_value: f32,
//         t_value: f32,
//         curve_closure: Box<dyn Fn(f32) -> f32>,
//     ) -> f32 {
//         let lerp_value = (curve_closure)(t_value);

//         (1. - lerp_value) * start_value + (lerp_value) * end_value
//     }
// }

// https://www.youtube.com/watch?v=syR0klfncCk
