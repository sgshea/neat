use std::collections::HashMap;

use crate::{
    genome::genome::Genome,
    nn::nn::{NetworkError, NeuralNetwork},
};

/// Continuous Time Recurrent Neural Network (CTRNN) implementation
pub struct CtrnnNetwork<'n> {
    genome: &'n Genome,

    // Node activations/states (current values for each neuron)
    states: Vec<f32>,

    // Time constants for each neuron (controls rate of change)
    time_constants: Vec<f32>,

    // Biases for each neuron
    biases: Vec<f32>,

    // Map from node ID to index in states array
    node_to_index: HashMap<usize, usize>,

    // Maps connections for faster evaluation
    connections: Vec<(usize, usize, f32)>, // (from_idx, to_idx, weight)

    // Default time step
    dt: f32,
}

impl<'n> NeuralNetwork<'n> for CtrnnNetwork<'n> {
    fn new(genome: &'n Genome) -> Result<Self, NetworkError> {
        let num_nodes = genome.nodes.len();

        // Create mapping from node IDs to indices
        let mut node_to_index = HashMap::with_capacity(num_nodes);
        for (i, &node_id) in genome.nodes.keys().enumerate() {
            node_to_index.insert(node_id, i);
        }

        // Initialize states and parameters
        let mut states = vec![0.0; num_nodes];
        let mut time_constants = vec![1.0; num_nodes];
        let mut biases = vec![0.0; num_nodes];

        // Set actual parameters
        for (&node_id, node) in &genome.nodes {
            if let Some(&idx) = node_to_index.get(&node_id) {
                time_constants[idx] = node.time_constant;
                biases[idx] = node.bias;
            }
        }

        // Set initial states for input nodes
        for &input_id in &genome.input_nodes {
            if let Some(&idx) = node_to_index.get(&input_id) {
                states[idx] = 0.0;
            }
        }

        // Preprocess connections for faster evaluation
        let mut connections = Vec::new();
        for conn in genome.connections.values() {
            if conn.enabled {
                if let (Some(&from_idx), Some(&to_idx)) = (
                    node_to_index.get(&conn.in_node),
                    node_to_index.get(&conn.out_node),
                ) {
                    connections.push((from_idx, to_idx, conn.weight));
                }
            }
        }

        Ok(CtrnnNetwork {
            genome,
            states,
            time_constants,
            biases,
            node_to_index,
            connections,
            dt: 0.1, // Default time step
        })
    }

    fn activate(&mut self, inputs: &[f32]) -> Result<Vec<f32>, NetworkError> {
        if inputs.len() != self.genome.input_nodes.len() {
            return Err(NetworkError::InvalidInput(
                "Number of inputs does not match network input size".into(),
            ));
        }

        // Set input node values
        for (i, &node_id) in self.genome.input_nodes.iter().enumerate() {
            if let Some(&idx) = self.node_to_index.get(&node_id) {
                self.states[idx] = inputs[i];
            }
        }

        // Set bias node value (always 1.0)
        if let Some(&idx) = self.node_to_index.get(&self.genome.bias_node) {
            self.states[idx] = 1.0;
        }

        // Perform CTRNN update step
        let mut next_states = self.states.clone();

        // Calculate derivatives
        for i in 0..self.states.len() {
            // Skip input nodes - they are set directly
            if self
                .genome
                .input_nodes
                .iter()
                .any(|&id| self.node_to_index.get(&id) == Some(&i))
            {
                continue;
            }

            // Calculate weighted input sum for this neuron
            let mut input_sum = self.biases[i];

            for &(from_idx, to_idx, weight) in &self.connections {
                if to_idx == i {
                    let from_activation = self.states[from_idx];
                    input_sum += from_activation * weight;
                }
            }

            // Get the activation function for this node
            let node_id = self
                .genome
                .nodes
                .keys()
                .find(|&&key| self.node_to_index.get(&key) == Some(&i))
                .unwrap();
            let node = &self.genome.nodes[node_id];

            // Apply activation function
            let target_activation = node.activation.activate(input_sum);

            // Calculate rate of change using time constant
            let dy = (target_activation - self.states[i]) / self.time_constants[i];

            // Euler integration step
            next_states[i] += dy * self.dt;
        }

        // Update states
        self.states = next_states;

        // Collect outputs
        let outputs: Vec<f32> = self
            .genome
            .output_nodes
            .iter()
            .filter_map(|&node_id| {
                self.node_to_index
                    .get(&node_id)
                    .map(|&idx| self.states[idx])
            })
            .collect();

        Ok(outputs)
    }
}

impl<'n> CtrnnNetwork<'n> {
    // Additional CTRNN-specific methods

    pub fn with_time_step(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    pub fn set_time_constants(&mut self, constants: &HashMap<usize, f32>) {
        for (node_id, constant) in constants {
            if let Some(&idx) = self.node_to_index.get(node_id) {
                self.time_constants[idx] = *constant;
            }
        }
    }

    pub fn set_biases(&mut self, biases: &HashMap<usize, f32>) {
        for (node_id, bias) in biases {
            if let Some(&idx) = self.node_to_index.get(node_id) {
                self.biases[idx] = *bias;
            }
        }
    }

    pub fn reset_states(&mut self) {
        self.states.fill(0.0);

        // Reset bias node
        if let Some(&idx) = self.node_to_index.get(&self.genome.bias_node) {
            self.states[idx] = 1.0;
        }
    }
}
