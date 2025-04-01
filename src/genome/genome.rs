use std::collections::{HashMap, HashSet};

use rand::{seq::IteratorRandom, Rng, RngCore};

use crate::{
    context::{ActivationFunction, NeatConfig},
    nn::nn::NetworkType,
    state::InnovationRecord,
};

use super::genes::{ConnectionGene, NodeGene};

// Genome is a single entity
#[derive(Debug, Clone)]
pub struct Genome {
    pub nodes: HashMap<usize, NodeGene>,
    pub connections: HashMap<usize, ConnectionGene>,
    pub connection_set: HashSet<(usize, usize)>,

    // Keys of input nodes
    pub input_nodes: Vec<usize>,
    pub bias_node: usize,
    // Keys of output nodes
    pub output_nodes: Vec<usize>,

    pub fitness: f32,
    pub adjusted_fitness: f32,
}

impl Genome {
    pub fn new() -> Self {
        Genome {
            nodes: HashMap::new(),
            connections: HashMap::new(),
            connection_set: HashSet::new(),
            input_nodes: Vec::new(),
            bias_node: 0,
            output_nodes: Vec::new(),
            fitness: 0.0,
            adjusted_fitness: 0.0,
        }
    }

    // Return a new genome from another, with fitness reset
    pub fn from_existing(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            connections: self.connections.clone(),
            connection_set: self.connection_set.clone(),
            input_nodes: self.input_nodes.clone(),
            bias_node: self.bias_node,
            output_nodes: self.output_nodes.clone(),
            fitness: 0.0,
            adjusted_fitness: 0.0,
        }
    }

    pub fn create_initial_genome(
        input_size: usize,
        output_size: usize,
        config: &NeatConfig,
        rng: &mut dyn RngCore,
        innovation: &mut InnovationRecord,
    ) -> Self {
        let mut nodes = HashMap::with_capacity(input_size + 1 + output_size);
        let mut input_nodes = Vec::with_capacity(input_size);
        let mut output_nodes = Vec::with_capacity(output_size);

        for _ in 0..input_size {
            let idx = innovation.record_node_innovation();
            nodes.insert(idx, NodeGene::new(idx, ActivationFunction::Identity));
            input_nodes.push(idx);
        }

        // Bias input node
        let bias_idx = innovation.record_node_innovation();
        let bias = NodeGene::new(bias_idx, ActivationFunction::Identity);
        nodes.insert(bias_idx, bias);

        for _ in 0..output_size {
            let idx = innovation.record_node_innovation();
            let mut node = NodeGene::new(idx, config.default_activation_function);
            if config.network_type == NetworkType::CTRNN {
                // randomize time constant and bias
                node.time_constant = rng.random_range(0.1..5.0);
                node.bias = rng.random_range(-1.0..1.0);
            }
            nodes.insert(idx, node);
            output_nodes.push(idx);
        }

        let mut connections = HashMap::with_capacity(input_size * output_size);
        let mut connection_set = HashSet::with_capacity(input_size * output_size);

        // Create initial connections between the input and output nodes
        for i in &input_nodes {
            for j in &output_nodes {
                let connection = (*i, *j);
                connection_set.insert(connection);
                let innovation = innovation.record_connection_innovation(*i, *j);
                connections.insert(
                    innovation,
                    ConnectionGene::new(connection, rng.random_range(-1.0..1.0), innovation),
                );
            }
        }
        // Connections from bias node to outputs
        for j in &output_nodes {
            let connection = (bias_idx, *j);
            connection_set.insert(connection);
            let innovation = innovation.record_connection_innovation(bias_idx, *j);
            connections.insert(
                innovation,
                ConnectionGene::new(connection, rng.random_range(-1.0..1.0), innovation),
            );
        }

        Self {
            nodes,
            connections,
            connection_set,
            input_nodes,
            bias_node: bias_idx,
            output_nodes,
            fitness: 0.0,
            adjusted_fitness: 0.0,
        }
    }

    pub fn mutate(
        &mut self,
        config: &NeatConfig,
        rng: &mut dyn RngCore,
        innovation_record: &mut InnovationRecord,
    ) {
        // Weight mutation
        if rng.random::<f32>() < config.weight_mutation_prob {
            for connection in &mut self.connections.values_mut() {
                if rng.random::<f32>() < config.weight_perturb_prob {
                    // Perturb weight slightly
                    connection.weight += rng.random_range(-0.5..0.5);
                } else {
                    // Assign completely new random weight
                    connection.weight = rng.random_range(-1.0..1.0);
                }
            }
        }

        // Add connection mutation
        if rng.random::<f32>() < config.new_connection_prob {
            self.add_connection_mutation(rng, innovation_record);
        }

        // Add node mutation
        if rng.random::<f32>() < config.new_node_prob {
            self.add_node_mutation(config, rng, innovation_record);
        }

        // Toggle enable/disable mutation
        if rng.random::<f32>() < config.toggle_enable_prob && !self.connections.is_empty() {
            let random_connection = self.connections.values_mut().choose(rng).unwrap();
            random_connection.enabled = !random_connection.enabled;
        }

        if config.network_type == NetworkType::CTRNN {
            self.mutate_node_parameters(config, rng);
        }
    }

    fn add_connection_mutation(
        &mut self,
        rng: &mut dyn RngCore,
        innovation: &mut InnovationRecord,
    ) {
        // Collect all possible node pairs that could be connected
        let mut possible_connections = Vec::new();

        // For efficiency, precompute which nodes are inputs/outputs
        let input_nodes: std::collections::HashSet<_> = self.input_nodes.iter().cloned().collect();
        let output_nodes: std::collections::HashSet<_> =
            self.output_nodes.iter().cloned().collect();

        // Try all possible node pairs
        for &from_node in self.nodes.keys() {
            // Skip output nodes as sources for connections
            if output_nodes.contains(&from_node) {
                continue;
            }

            for &to_node in self.nodes.keys() {
                // Skip input nodes as targets for connections
                if input_nodes.contains(&to_node) {
                    continue;
                }

                // Skip connections that already exist
                if self.connection_set.contains(&(from_node, to_node)) {
                    continue;
                }

                // Skip self-connections
                if from_node == to_node {
                    continue;
                }

                // This is a valid potential connection
                possible_connections.push((from_node, to_node));
            }
        }

        // If we found at least one valid connection, randomly choose one
        if !possible_connections.is_empty() {
            let (from_node, to_node) =
                possible_connections[rng.random_range(0..possible_connections.len())];

            // Get innovation number for this connection - consistent across population
            let innovation_number = innovation.record_connection_innovation(from_node, to_node);

            // If we already have this connection (possible with crossover), skip
            if self.connections.contains_key(&innovation_number) {
                return;
            }

            // Create and add the connection
            let connection = ConnectionGene {
                in_node: from_node,
                out_node: to_node,
                weight: rng.random_range(-1.0..1.0),
                enabled: true,
                innovation: innovation_number,
            };

            self.connections.insert(innovation_number, connection);
            self.connection_set.insert((from_node, to_node));
        }
    }

    // Helper method for add node mutation
    fn add_node_mutation(
        &mut self,
        config: &NeatConfig,
        rng: &mut dyn RngCore,
        innovation_record: &mut InnovationRecord,
    ) {
        // Can't add node if there are no connections
        if self.connections.is_empty() {
            return;
        }

        // Select a random connection
        let innovation = self.connections.keys().cloned().choose(rng).unwrap();
        let connection = self.connections.get(&innovation).unwrap().clone();

        // Only split enabled connections
        if !connection.enabled {
            return;
        }

        // Disable the selected connection
        self.connections.get_mut(&innovation).unwrap().enabled = false;

        // Get consistent innovation numbers for this split
        let (new_node_id, in_conn_innovation, out_conn_innovation) = innovation_record
            .record_node_split(innovation, connection.in_node, connection.out_node);

        // Check if the node already exists (could happen in crossover)
        if !self.nodes.contains_key(&new_node_id) {
            // Create a new node
            let mut new_node = NodeGene::new(new_node_id, config.default_activation_function);
            if config.network_type == NetworkType::CTRNN {
                // randomize time constant and bias
                new_node.time_constant = rng.random_range(0.1..5.0);
                new_node.bias = rng.random_range(-1.0..1.0);
            }
            self.nodes.insert(new_node_id, new_node);
        }

        // Create two new connections with the appropriate innovation numbers
        let in_conn = ConnectionGene::new(
            (connection.in_node, new_node_id),
            1.0, // Weight from input to new node is 1.0
            in_conn_innovation,
        );

        let out_conn = ConnectionGene::new(
            (new_node_id, connection.out_node),
            connection.weight, // Weight from new node to output is the original weight
            out_conn_innovation,
        );

        // Add the new connections
        self.connections.insert(in_conn.innovation, in_conn);
        self.connections.insert(out_conn.innovation, out_conn);

        // Update connection set
        self.connection_set
            .insert((connection.in_node, new_node_id));
        self.connection_set
            .insert((new_node_id, connection.out_node));
    }

    // Mutate node bias and time constant (primarily for CTRNN)
    fn mutate_node_parameters(&mut self, config: &NeatConfig, rng: &mut dyn RngCore) {
        for node in self.nodes.values_mut() {
            // Skip input nodes for bias mutation (they typically don't biases)
            let is_input = self.input_nodes.contains(&node.id) || node.id == self.bias_node;

            // Mutate bias (except for input nodes)
            if !is_input && rng.random::<f32>() < config.bias_mutation_prob {
                if rng.random::<f32>() < config.param_perturb_prob {
                    // Perturb existing bias
                    node.bias += rng.random_range(-0.5..0.5);
                    node.bias = node.bias.clamp(-8.0, 8.0);
                } else {
                    // Assign new random bias
                    node.bias = rng.random_range(-1.0..1.0);
                }
            }

            // Mutate time constant (for CTRNN)
            if rng.random::<f32>() < config.time_constant_mutation_prob {
                if rng.random::<f32>() < config.param_perturb_prob {
                    // Perturb existing time constant
                    let delta = rng.random_range(-0.1..0.1);
                    node.time_constant = (node.time_constant + delta).max(0.1);
                } else {
                    // Assign new random time constant
                    // Values between 0.1 (fast) and 5.0 (slow)
                    node.time_constant = rng.random_range(0.1..5.0);
                }
            }
        }
    }

    pub fn compatibility_distance(&self, other: &Genome, config: &NeatConfig) -> f32 {
        let mut num_excess = 0;
        let mut num_disjoint = 0;
        let mut weight_diff_sum = 0.0;
        let mut num_matching = 0;

        // Find the highest innovation number in each genome
        let max_innov_self = self.connections.keys().max().copied().unwrap_or(0);
        let max_innov_other = other.connections.keys().max().copied().unwrap_or(0);
        let max_genome_size = self.connections.len().max(other.connections.len());

        // If genomes are empty, they're identical
        if max_genome_size == 0 {
            return 0.0;
        }

        // Normalize by size of larger genome
        let size_normalization = if max_genome_size < 20 {
            1.0
        } else {
            max_genome_size as f32
        };

        // Compare connections
        let mut all_innovations = HashSet::new();
        self.connections.keys().for_each(|&k| {
            all_innovations.insert(k);
        });
        other.connections.keys().for_each(|&k| {
            all_innovations.insert(k);
        });

        for &innov in &all_innovations {
            match (self.connections.get(&innov), other.connections.get(&innov)) {
                (Some(gene1), Some(gene2)) => {
                    // Matching genes
                    num_matching += 1;
                    weight_diff_sum += (gene1.weight - gene2.weight).abs();
                }
                (Some(_), None) => {
                    // Gene in self but not in other
                    if innov > max_innov_other {
                        num_excess += 1;
                    } else {
                        num_disjoint += 1;
                    }
                }
                (None, Some(_)) => {
                    // Gene in other but not in self
                    if innov > max_innov_self {
                        num_excess += 1;
                    } else {
                        num_disjoint += 1;
                    }
                }
                (None, None) => unreachable!(),
            }
        }

        // Calculate average weight difference
        let avg_weight_diff = if num_matching > 0 {
            weight_diff_sum / num_matching as f32
        } else {
            0.0
        };

        // Calculate and return compatibility distance
        (config.compatibility_disjoint_coefficient * num_disjoint as f32) / size_normalization
            + (config.compatibility_disjoint_coefficient * num_excess as f32) / size_normalization
            + (config.compatibility_weight_coefficient * avg_weight_diff)
    }

    pub fn crossover(&self, other: &Genome, rng: &mut dyn RngCore) -> Genome {
        let mut child = self.from_existing();

        // Copy input and output nodes
        child.input_nodes = self.input_nodes.clone();
        child.output_nodes = self.output_nodes.clone();
        child.bias_node = self.bias_node;

        // Determine which parent is more fit
        let (more_fit, less_fit) = if self.fitness > other.fitness {
            (self, other)
        } else if other.fitness > self.fitness {
            (other, self)
        } else {
            if rng.random_bool(0.5) {
                (self, other)
            } else {
                (other, self)
            }
        };

        // Add all nodes from both parents
        for (node_id, node) in &more_fit.nodes {
            child.nodes.insert(*node_id, node.clone());
        }
        for (node_id, node) in &less_fit.nodes {
            if !child.nodes.contains_key(node_id) {
                child.nodes.insert(*node_id, node.clone());
            }
        }

        // Sort innovations to add connections in a deterministic order
        let mut all_innovations: Vec<usize> = more_fit
            .connections
            .keys()
            .chain(less_fit.connections.keys())
            .cloned()
            .collect();
        all_innovations.sort(); // Add in consistent order

        // Handle connections
        for &innov in &all_innovations {
            match (
                more_fit.connections.get(&innov),
                less_fit.connections.get(&innov),
            ) {
                (Some(gene1), Some(gene2)) => {
                    // Matching genes - inherit randomly
                    let chosen_gene = if rng.random_bool(0.5) { gene1 } else { gene2 };

                    if !child
                        .connection_set
                        .contains(&(chosen_gene.in_node, chosen_gene.out_node))
                    {
                        child.connections.insert(innov, chosen_gene.clone());
                        child
                            .connection_set
                            .insert((chosen_gene.in_node, chosen_gene.out_node));
                    }
                }
                (Some(gene), None) | (None, Some(gene)) => {
                    // Disjoint or excess gene - inherit from more fit parent
                    if !child
                        .connection_set
                        .contains(&(gene.in_node, gene.out_node))
                    {
                        child.connections.insert(innov, gene.clone());
                        child.connection_set.insert((gene.in_node, gene.out_node));
                    }
                }
                (None, None) => unreachable!(),
            }
        }

        child
    }

    // Gets a fitness penalty based on complexity of genome structure
    pub fn apply_parsimony_pressure(&self, config: &NeatConfig, original_fitness: f32) -> f32 {
        // Skip penalty if bad fitness
        if original_fitness <= 0.0 {
            return original_fitness;
        }

        // Count non-input/output nodes (hidden nodes)
        let num_hidden_nodes =
            self.nodes.len() - (self.input_nodes.len() + self.output_nodes.len());

        // Skip penalty if complexity is below threshold
        if num_hidden_nodes <= config.complexity_threshold {
            return original_fitness;
        }

        // Calculate excess complexity
        let excess_nodes = num_hidden_nodes.saturating_sub(config.target_complexity);

        // Calculate connection penalty
        let connection_penalty =
            config.connections_penalty_coefficient * self.connections.len() as f32;

        // Node penalty increases quadratically with excess
        let node_penalty = if excess_nodes > 0 {
            config.complexity_penalty_coefficient * (excess_nodes as f32).powf(1.5)
        } else {
            0.0
        };

        // Apply combined penalty
        let penalized_fitness = original_fitness - node_penalty - connection_penalty;
        penalized_fitness.max(0.00001) // Prevent zero fitness
    }
}
