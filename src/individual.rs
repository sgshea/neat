use crate::genome::Genome;
use crate::innovation_record::InnovationRecord;

pub struct Individual {
    pub genome: Genome,
    pub fitness: f32,
}

impl Individual {
    pub fn new(input_nodes: usize, output_nodes: usize, hidden_nodes: usize, innovation_record: &mut InnovationRecord) -> Self {
        Self {
            genome: Genome::new(input_nodes, output_nodes, hidden_nodes, innovation_record),
            fitness: 0.0,
        }
    }
}