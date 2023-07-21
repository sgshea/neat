pub struct InnovationRecord {
    node_amount: usize,
    connection_amount: usize,
    species_amount: usize,
}

impl InnovationRecord {
    pub fn new(initial_amount: usize) -> InnovationRecord {
        InnovationRecord {
            node_amount: initial_amount,
            connection_amount: 0,
            species_amount: 0,
        }
    }

    pub fn new_node(&mut self) -> usize {
        let id = self.node_amount;
        self.node_amount += 1;
        id
    }

    pub fn new_connection(&mut self) -> usize {
        let id = self.connection_amount;
        self.connection_amount += 1;
        id
    }

    // Sometimes needed when checking for cyclic graph
    pub fn remove_last_connection(&mut self) {
        self.connection_amount -= 1;
    }

    pub fn new_species(&mut self) -> usize {
        let id = self.species_amount;
        self.species_amount += 1;
        id
    }
}
