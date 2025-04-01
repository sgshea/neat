use std::collections::HashMap;

// Manages the amount of species through adjusting the compatibility threshold
// Also holds the current species counter (simple id for species)
#[derive(Debug, Clone)]
pub struct SpeciationManager {
    // Changing compatability threshold
    pub compatibility_threshold: f32,
    // Incrementing species id
    species_counter: usize,
    // Amount of species we should try for
    target_species_count: usize,
}

impl SpeciationManager {
    pub fn new(
        compatibility_threshold: f32,
        current_species_count: usize,
        target_species_count: usize,
    ) -> Self {
        SpeciationManager {
            compatibility_threshold,
            species_counter: current_species_count,
            target_species_count,
        }
    }

    // Gives a new species id
    pub fn new_species(&mut self) -> usize {
        self.species_counter += 1;
        self.species_counter
    }

    // Adjusts the compatibility threshold based on the current species count against the target species count
    pub fn adjust_threshold(&mut self, current_species_count: usize) {
        if current_species_count > self.target_species_count * 2 {
            self.compatibility_threshold *= 1.3;
        } else if current_species_count < self.target_species_count / 2 {
            self.compatibility_threshold *= 0.95;
        }
    }
}

// Keeps track of node and connection innovations
#[derive(Debug, Clone)]
pub struct InnovationRecord {
    // Keeps track of node innovations
    node_innovation_counter: usize,
    // Keeps track of connection innovations
    connection_innovation_counter: usize,

    // Key: (in_node_id, out_node_id) -> innovation_id
    connection_innovations: HashMap<(usize, usize), usize>,

    // Tracking node splits
    // Key: (connection_id) -> Value: (node_id, in, out)
    node_splits: HashMap<usize, (usize, usize, usize)>,
}

impl InnovationRecord {
    pub fn new() -> Self {
        InnovationRecord {
            node_innovation_counter: 0,
            connection_innovation_counter: 0,
            connection_innovations: HashMap::new(),
            node_splits: HashMap::new(),
        }
    }

    pub fn record_node_innovation(&mut self) -> usize {
        let innovation = self.node_innovation_counter;
        self.node_innovation_counter += 1;
        innovation
    }

    // Gets existing innovation or creates new one
    pub fn record_connection_innovation(&mut self, in_node: usize, out_node: usize) -> usize {
        let connection = (in_node, out_node);

        *self
            .connection_innovations
            .entry(connection)
            .or_insert_with(|| {
                let innovation = self.connection_innovation_counter;
                self.connection_innovation_counter += 1;
                innovation
            })
    }

    // Records the connection split with new node
    pub fn record_node_split(
        &mut self,
        connection_id: usize,
        in_node: usize,
        out_node: usize,
    ) -> (usize, usize, usize) {
        // Check if this exact split happened before
        if let Some(&result) = self.node_splits.get(&connection_id) {
            return result;
        }

        // Create a new node
        let new_node_id = self.node_innovation_counter;
        self.node_innovation_counter += 1;

        // Get innovations for the two new connections
        let in_to_new = self.record_connection_innovation(in_node, new_node_id);
        let new_to_out = self.record_connection_innovation(new_node_id, out_node);

        // Record split
        let result = (new_node_id, in_to_new, new_to_out);
        self.node_splits.insert(connection_id, result);

        result
    }
}
