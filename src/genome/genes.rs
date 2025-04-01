use crate::context::ActivationFunction;

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

    // CTRNN values
    pub bias: f32,
    pub time_constant: f32,
}

impl NodeGene {
    pub fn new(id: usize, activation: ActivationFunction) -> Self {
        NodeGene {
            id,
            activation,
            bias: 0.0,
            time_constant: 1.0,
        }
    }

    // Runs activation function on input + bias
    pub fn activate(&self, input: &[f32]) -> f32 {
        let sum = input.iter().sum::<f32>();
        self.activation.activate(sum)
    }
}
