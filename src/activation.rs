#[derive(Clone, Copy)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
}

impl Activation {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + std::f32::consts::E.powf(-x)),
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
            Activation::Linear => x,
        }
    }
}
