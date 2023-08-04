use crate::activation::Activation;
use crate::genome::Genome;
use crate::innovation_record::InnovationRecord;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Individual {
    genome: Genome,
    pub fitness: f64,
}

impl Individual {
    pub fn new(
        input_nodes: usize,
        output_nodes: usize,
        hidden_nodes: usize,
        innovation_record: &mut InnovationRecord,
    ) -> Self {
        Self {
            genome: Genome::new(input_nodes, output_nodes, hidden_nodes, innovation_record),
            fitness: 0.0,
        }
    }

    // Return mutated individual
    pub fn mutate(&self, innovation_record: &mut InnovationRecord) -> Individual {
        let mut new_individual = self.clone();
        new_individual.genome.mutate(innovation_record);
        new_individual
    }

    // Mutate genome in place
    pub fn mutate_in_place(&mut self, innovation_record: &mut InnovationRecord) {
        self.genome.mutate(innovation_record);
    }

    // Crossover with another individual to return offspring
    pub fn crossover(&self, other: &Individual) -> Individual {
        let mut new_individual = self.clone();
        new_individual.genome.crossover(&other.genome);
        new_individual
    }

    // Return compatibility distance between two individuals
    pub fn distance(&self, other: &Individual) -> f64 {
        self.genome.compatability_distance(&other.genome)
    }

    pub fn activate(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.genome.output(&*inputs, Activation::Sigmoid)
    }

    pub fn output_graph(&self) {
        self.genome.output_graph()
    }
}

impl Eq for Individual {}

impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl PartialOrd for Individual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Individual {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .fitness
            .partial_cmp(&self.fitness)
            .expect("Failed to compare fitness")
    }
}
