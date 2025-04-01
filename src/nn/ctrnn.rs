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
        // Set inputs and bias
        self.set_inputs(inputs)?;

        // Temporary array for next states
        let mut next_states = self.states.clone();

        // Update all non-input neurons
        for i in 0..self.states.len() {
            // Skip input nodes (including bias)
            if self.is_input_node(i) {
                continue;
            }

            // Calculate total input to this neuron
            let mut weighted_input = self.biases[i];

            // Sum all incoming connections
            for &(from_idx, to_idx, weight) in &self.connections {
                if to_idx == i {
                    // The key CTRNN part: apply sigmoid to source states
                    let from_state_activation = sigmoid(self.states[from_idx]);
                    weighted_input += from_state_activation * weight;
                }
            }

            // CTRNN differential equation: τ(dy/dt) = -y + I
            let tau = self.time_constants[i];
            let dy_dt = (-self.states[i] + weighted_input) / tau;

            // Euler integration
            next_states[i] = self.states[i] + dy_dt * self.dt;
        }

        // Update states
        self.states = next_states;

        // Apply sigmoid to output states when reading them
        let outputs = self
            .genome
            .output_nodes
            .iter()
            .filter_map(|&id| {
                self.node_to_index
                    .get(&id)
                    .map(|&idx| sigmoid(self.states[idx]))
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

    fn set_inputs(&mut self, inputs: &[f32]) -> Result<(), NetworkError> {
        if inputs.len() != self.genome.input_nodes.len() {
            return Err(NetworkError::InvalidInput("Input size not correct.".into()));
        }

        for (idx, &input) in inputs.iter().enumerate() {
            self.states[self.node_to_index[&self.genome.input_nodes[idx]]] = input;
        }

        // Set bias to 1.0
        if let Some(&idx) = self.node_to_index.get(&self.genome.bias_node) {
            self.states[idx] = 1.0;
        }

        Ok(())
    }

    pub fn reset_states(&mut self) {
        self.states.fill(0.0);

        // Reset bias node
        if let Some(&idx) = self.node_to_index.get(&self.genome.bias_node) {
            self.states[idx] = 1.0;
        }
    }

    // Helper to check if a node is an input
    fn is_input_node(&self, idx: usize) -> bool {
        self.genome
            .input_nodes
            .iter()
            .any(|&id| self.node_to_index.get(&id) == Some(&idx))
            || self.node_to_index.get(&self.genome.bias_node) == Some(&idx)
    }
}

// Helper function to compute sigmoid activation
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
