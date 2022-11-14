use cgmath::{Vector3, Zero};
use typed_builder::TypedBuilder;

fn sample_lookup_table(t_value: f32, curve_closure: Box<dyn Fn(f32) -> f32>) -> f32 {
    curve_closure(t_value)
}

pub const QUADRATIC_CENTERED: fn(f32) -> f32 = |x: f32| -4. * (x) * (x - 1.);

pub struct Curve 
{
    point_list: Vec<CurvePoint>,
    closure: Box<fn(f32) -> f32>,
}

type Colour3 = Vector3<f32>;
type Vec3 = Vector3<f32>;

impl Curve {
    pub fn vec3_curve(
        start_value: Vec3,
        end_value: Vec3,
        t_value: f32,
        curve_closure: Box<dyn Fn(f32) -> f32>,
    ) -> Vec3 {
        let lerp_value = sample_lookup_table(t_value, curve_closure);

        (1. - lerp_value) * start_value + (lerp_value) * end_value
    }

    pub fn float_curve(
        start_value: f32,
        end_value: f32,
        t_value: f32,
        curve_closure: Box<dyn Fn(f32) -> f32>,
    ) -> f32 {
        let lerp_value = sample_lookup_table(t_value, curve_closure);

        (1. - lerp_value) * start_value + (lerp_value) * end_value
    }
}

pub struct CurvePoint{
    key: f32,
    value: f32,
}

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

#[derive(TypedBuilder)]
pub struct Particle {
    #[builder(default = Vec3::zero())]
    curr_position: Vec3,
    #[builder(default = Vec3::zero())]
    curr_velocity: Vec3,
    #[builder(default = Vec3::zero())]
    curr_force: Vec3,
    force_curve: Curve,
    #[builder(default = 1.0)]
    curr_size: f32,
    #[builder(default = Colour3::zero())]
    curr_color: Colour3,
    #[builder(default = 1.0)]
    curr_transparency: f32,
    #[builder(default = 0.0)]
    curr_lifetime: f32,
    #[builder(default = 1.0)]
    total_lifetime: f32,
    // shape?
}

impl Particle 
{
    fn update(&mut self, dt: f32) {
        self.curr_position += self.curr_velocity * dt;
        self.curr_velocity += self.curr_force * dt;
        self.curr_force = Vec3::zero();

        self.update_curve_values();
    }

    fn update_curve_values(&mut self) {
        let normalized_time = self.curr_lifetime / self.total_lifetime;

        self.curr_position.y = QUADRATIC_CENTERED(normalized_time);
    }

    fn get_mesh(&self, device: wgpu::Device) -> crate::model::Mesh {
        let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(QUAD_VERTS),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );

        let index_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &device,
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
}

trait PhysicsPoint {
    fn apply_force(&mut self, force: Vec3);
}

impl PhysicsPoint for Particle {
    fn apply_force(&mut self, force: Vec3) {
        self.curr_force += force;
    }
}

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
