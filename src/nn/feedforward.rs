use std::collections::{HashMap, HashSet, VecDeque};

use crate::genome::genome::Genome;

use super::nn::{NetworkError, NeuralNetwork};

/// Simple feedforward neural network implementation
pub struct FeedforwardNetwork<'n> {
    genome: &'n Genome,

    // Represents the topological sorting of the nodes in priority order
    sorted_nodes: Vec<usize>,

    // Map from node ID to index in the outputs array
    node_to_index: HashMap<usize, usize>,

    // Tracks which connections are used in the feedforward network
    used_connections: HashSet<(usize, usize)>,
}

impl<'n> NeuralNetwork<'n> for FeedforwardNetwork<'n> {
    /// Create a new feedforward network by borrowing the genome
    /// Ignores connections that would create cycles
    fn new(genome: &'n Genome) -> Result<Self, NetworkError> {
        // Create a mapping from node IDs to sequential indices
        let mut node_to_index = HashMap::new();
        for (i, &node_id) in genome.nodes.keys().enumerate() {
            node_to_index.insert(node_id, i);
        }

        // Track which connections are used in the feedforward network
        let mut used_connections = HashSet::new();

        // Initialize data structures for topological sort
        let mut in_degree = HashMap::new();
        let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();

        // Create an adjacency list from enabled connections
        let mut adjacency_list: HashMap<usize, Vec<usize>> = HashMap::new();
        for &node_id in genome.nodes.keys() {
            adjacency_list.insert(node_id, Vec::new());
            in_degree.insert(node_id, 0);
            outgoing.insert(node_id, Vec::new());
        }

        // Build the graph from enabled connections
        for conn in genome.connections.values() {
            if conn.enabled {
                adjacency_list
                    .get_mut(&conn.in_node)
                    .unwrap()
                    .push(conn.out_node);
            }
        }

        // Detect and remove cycles
        // First, create a working copy of the adjacency list to modify
        let mut working_graph = adjacency_list.clone();

        // Calculate initial in-degrees for each node
        for connections in working_graph.values() {
            for &out_node in connections {
                *in_degree.entry(out_node).or_insert(0) += 1;
            }
        }

        // Start with nodes that have no incoming edges
        let mut queue = VecDeque::new();
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        // Perform the topological sort
        let mut sorted_nodes = Vec::new();

        while let Some(node) = queue.pop_front() {
            sorted_nodes.push(node);

            // Get a copy of the edges to avoid mutable borrow issues
            let edges = working_graph.get(&node).unwrap().clone();

            // Process outgoing edges
            for &next in &edges {
                // Mark this connection as used in the feedforward network
                used_connections.insert((node, next));

                // Reduce in-degree of the target node
                if let Some(degree) = in_degree.get_mut(&next) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(next);
                    }
                }
            }

            // Clear the outgoing edges
            working_graph.get_mut(&node).unwrap().clear();
        }

        // If some nodes weren't visited, they're part of cycles and won't be in sorted_nodes
        // This implementation ignores those connections rather than returning an error

        Ok(FeedforwardNetwork {
            genome,
            sorted_nodes,
            node_to_index,
            used_connections,
        })
    }

    fn activate(&mut self, inputs: &[f32]) -> Result<Vec<f32>, NetworkError> {
        if inputs.len() != self.genome.input_nodes.len() {
            return Err(NetworkError::InvalidInput(
                "Number of inputs is not correct".into(),
            ));
        }

        // Outputs for each node - indexed by the mapping
        let mut outputs: Vec<f32> = vec![0.0; self.genome.nodes.len()];

        // Set input nodes
        for (i, &node_id) in self.genome.input_nodes.iter().enumerate() {
            if let Some(&idx) = self.node_to_index.get(&node_id) {
                outputs[idx] = inputs[i];
            }
        }

        // Process all nodes using sorted order
        for &node_id in &self.sorted_nodes {
            // Skip input nodes, already set
            if self.genome.input_nodes.contains(&node_id) {
                continue;
            }

            // Find all incoming connections and calculate weighted sum
            let mut weighted_inputs: Vec<f32> = Vec::new();

            for conn in self.genome.connections.values() {
                // Only use connections that are enabled AND part of the feedforward network
                if conn.out_node == node_id
                    && conn.enabled
                    && self
                        .used_connections
                        .contains(&(conn.in_node, conn.out_node))
                {
                    if let (Some(&in_idx), Some(&_out_idx)) = (
                        self.node_to_index.get(&conn.in_node),
                        self.node_to_index.get(&conn.out_node),
                    ) {
                        weighted_inputs.push(outputs[in_idx] * conn.weight);
                    }
                }
            }

            // Apply node activation function
            let node = &self.genome.nodes[&node_id];
            if let Some(&idx) = self.node_to_index.get(&node_id) {
                outputs[idx] = node.activate(&weighted_inputs);
            }
        }

        // Collect and return outputs
        Ok(self
            .genome
            .output_nodes
            .iter()
            .filter_map(|&node_id| self.node_to_index.get(&node_id).map(|&idx| outputs[idx]))
            .collect())
    }
}
