use cgmath::{Vector3};

pub trait Curve<T> {
    fn sample_curve(
        start_value: T,
        end_value: T,
        t_value: f32,
        curve_closure: Box<dyn Fn(f32) -> f32>,
    ) -> T;

    fn sample_LUT(t_value: f32, curve_closure: Box<dyn Fn(f32) -> f32>) -> f32 {
        curve_closure(t_value)
    }
}

type Vec3 = Vector3<f32>;
type Colour3 = Vector3<f32>;

pub struct Particle {
    position: Vec3,
    velocity: Vec3,
    size: f32,
    color: Colour3,
    transparency: f32,
    // shape?
    lifetime: f32,
}
