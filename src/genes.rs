use rand::Rng;

#[derive(Clone, Debug, Copy)]
pub struct ConnectionGene {
    pub innovation: usize,
    pub in_node: usize,
    pub out_node: usize,
    pub weight: f64,
    pub enabled: bool,
    pub is_recurrent: bool,
}

impl ConnectionGene {
    pub fn new(in_node: usize, out_node: usize, weight: f64, innovation: usize) -> Self {
        Self {
            in_node,
            out_node,
            weight,
            enabled: true,
            innovation,
            is_recurrent: false,
        }
    }

    pub fn mutate_weight(&mut self) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < 0.1 {
            self.weight = rng.gen_range(-5.0..5.0);
        } else {
            // add/subtract 20%
            self.weight += self.weight * rng.gen_range(-0.2..0.2);
        }
    }
}

#[derive(Clone, Debug)]
pub struct NodeGene {
    pub id: usize,
    pub node_type: NodeType,
    pub node_layer: usize,
    pub sum_inputs: f64,
    pub sum_outputs: f64,
}

impl NodeGene {
    pub fn new(
        id: usize,
        node_type: NodeType,
        node_layer: usize,
        sum_inputs: f64,
        sum_outputs: f64,
    ) -> Self {
        Self {
            id,
            node_type,
            node_layer,
            sum_inputs,
            sum_outputs,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    Bias,
    Input,
    Output,
    Hidden,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ActivationFunction {
    None,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
}

impl ActivationFunction {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::None => x,
            ActivationFunction::Sigmoid => 1.0 / (1.0 + std::f64::consts::E.powf(-x)),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU => x.max(0.01 * x),
        }
    }
}
