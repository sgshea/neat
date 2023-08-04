use crate::activation::Activation;

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Input,
    Output,
    Hidden,
    Bias,
}

#[derive(Debug, Clone)]
pub struct Node {
    node_id: usize,
    node_type: NodeType,
    sum_inputs: f32,
    output: f32,
}

impl Node {
    pub fn new(node_identifier: usize, node_type: NodeType) -> Node {
        Node {
            node_id: node_identifier,
            node_type,
            sum_inputs: 0.0,
            output: 0.0,
        }
    }

    pub fn update_sum(&mut self, sum_inputs: f32) {
        self.sum_inputs = sum_inputs;
    }

    pub fn activate(&mut self, func: Activation) -> Option<f32> {
        if self.node_type == NodeType::Input || self.node_type == NodeType::Bias {
            // Pass through
            self.output = self.sum_inputs;
            Some(self.output)
        } else if self.node_type == NodeType::Hidden || self.node_type == NodeType::Output {
            self.output = func.activate(self.sum_inputs);
            Some(self.output)
        } else {
            None
        }
    }

    pub fn get_type(&self) -> NodeType {
        self.node_type.clone()
    }

    pub fn get_id(&self) -> usize {
        self.node_id
    }

    pub fn get_sum(&self) -> f32 {
        self.sum_inputs
    }

    pub fn get_output(&self) -> f32 {
        self.output
    }
}
