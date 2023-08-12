use std::collections::HashMap;

pub struct InnovationRecord {
    // Innovation number stored as a hashmap of (from, to) -> innovation
    pub innovation_number: HashMap<(usize, usize), usize>,
    pub num_nodes: usize,
}

impl InnovationRecord {

    pub fn new() -> Self {
        InnovationRecord {
            innovation_number: HashMap::new(),
            num_nodes: 0,
        }
    }

    pub fn has_innovation(&self, from: usize, to: usize) -> bool {
        self.innovation_number.contains_key(&(from, to))
    }

    // Returns id of innovation
    // If innovation already exists, returns existing innovation
    pub fn new_innovation(&mut self, from: usize, to: usize) -> usize {
        let innovation = self.innovation_number.get(&(from, to));
        match innovation {
            Some(innovation) => *innovation,
            None => {
                let innovation = self.innovation_number.len();
                self.innovation_number.insert((from, to), innovation);
                innovation
            }
        }
    }

    pub fn new_node_innovation(&mut self) -> usize {
        let innovation = self.num_nodes;
        self.num_nodes += 1;
        innovation
    }
}
