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

#[derive(Debug, Clone)]
pub struct InnovationRecord {
    node_innovation_counter: usize,
    innovation_counter: usize,
    innovations: HashMap<(usize, usize), usize>,
}

impl InnovationRecord {
    pub fn new() -> Self {
        InnovationRecord {
            node_innovation_counter: 0,
            innovation_counter: 0,
            innovations: HashMap::new(),
        }
    }

    pub fn get_innovation_id(&self, innovation: (usize, usize)) -> Option<usize> {
        self.innovations.get(&innovation).copied()
    }

    pub fn record_innovation(&mut self, innovation: (usize, usize)) -> usize {
        let innovation_id = self.innovation_counter;
        self.innovation_counter += 1;
        self.innovations.insert(innovation, innovation_id);
        innovation_id
    }

    pub fn record_node_innovation(&mut self) -> usize {
        let innovation_id = self.node_innovation_counter;
        self.node_innovation_counter += 1;
        innovation_id
    }
}
