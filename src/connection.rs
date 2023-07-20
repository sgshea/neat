use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Connection {
    // input, output nodes handled by graph structure

    // unique innovation id
    innovation_id: usize,

    // weight of connection
    connection_weight: f32,

    // is connection enabled?
    // Not used right now, we just remove the connection
    enabled: bool,
}

impl Connection {
    pub fn new(innovation_id: usize, connection_weight: f32) -> Connection {
        Connection {
            innovation_id,
            connection_weight,
            enabled: true,
        }
    }

    pub fn set_disabled(&mut self) {
        self.enabled = false;
    }

    pub fn set_enabled(&mut self) {
        self.enabled = true;
    }

    pub fn swap_enabled(&mut self) {
        self.enabled = !self.enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn get_innovation_id(&self) -> usize {
        self.innovation_id
    }

    pub fn get_weight(&self) -> f32 {
        self.connection_weight
    }

    pub fn set_weight(&mut self, weight: f32) {
        self.connection_weight = weight;
    }
}
