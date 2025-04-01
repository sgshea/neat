#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConnectionGene {
    pub weight: f32,
    pub enabled: bool,

    // In and out nodes
    pub in_node: usize,
    pub out_node: usize,

    pub innovation: usize,
}

impl ConnectionGene {
    pub fn new(connection: (usize, usize), weight: f32, innovation: usize) -> Self {
        ConnectionGene {
            weight,
            enabled: true,
            in_node: connection.0,
            out_node: connection.1,
            innovation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeGene {
    pub id: usize,
    pub activation: ActivationFunction,
}

impl NodeGene {
    pub fn new(id: usize, activation: ActivationFunction) -> Self {
        NodeGene { id, activation }
    }

    // Runs activation function on input + bias
    pub fn activate(&self, input: &[f32]) -> f32 {
        let sum = input.iter().sum::<f32>();
        self.activation.activate(sum)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFunction {
    Identity,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
}

impl ActivationFunction {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Identity => x,
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Relu => x.max(0.0),
            ActivationFunction::LeakyRelu => x.max(0.01 * x),
        }
    }
}
