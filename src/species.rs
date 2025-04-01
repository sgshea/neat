use crate::{context::NeatConfig, genome::genome::Genome, state::InnovationRecord};
use rand::{seq::IndexedRandom, Rng, RngCore};

#[derive(Debug, Clone)]
pub struct Species {
    pub id: usize,
    pub genomes: Vec<Genome>,
    pub representative: Genome,
    pub staleness: usize,
    pub best_fitness: f32,
    pub best_fitness_genome: Option<Genome>,
    pub average_fitness: f32,
}

impl Species {
    pub fn new(id: usize, representative: Genome) -> Self {
        Species {
            id,
            genomes: vec![],
            representative,
            staleness: 0,
            best_fitness: 0.0,
            best_fitness_genome: None,
            average_fitness: 0.0,
        }
    }

    pub fn add(&mut self, genome: Genome) {
        self.genomes.push(genome);
    }

    pub fn calculate_average_fitness(&mut self) {
        if self.genomes.is_empty() {
            self.average_fitness = 0.0;
            return;
        }

        let total_fitness: f32 = self.genomes.iter().map(|g| g.fitness).sum();
        self.average_fitness = total_fitness / self.genomes.len() as f32;
    }

    pub fn update_best_fitness(&mut self) -> bool {
        let best_genome = match self.genomes.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Some(genome) => genome,
            None => return false,
        };

        if best_genome.fitness > self.best_fitness {
            self.best_fitness = best_genome.fitness;
            self.best_fitness_genome = Some(best_genome.clone());
            self.representative = best_genome.clone();
            self.staleness = 0;
            true
        } else {
            self.staleness += 1;
            false
        }
    }

    pub fn select_representative(&self) -> Genome {
        self.genomes.choose(&mut rand::rng()).unwrap().clone()
    }

    pub fn make_child(
        &self,
        config: &NeatConfig,
        rng: &mut dyn RngCore,
        innovation: &mut InnovationRecord,
    ) -> Genome {
        let mut child = if rng.random::<f32>() < config.crossover_rate {
            // Crossover
            let parent1 = self.genomes.choose(rng).unwrap();
            let parent2 = self.genomes.choose(rng).unwrap();
            Genome::crossover(parent1, parent2, rng)
        } else {
            // Mutation
            let mut parent = self.genomes.choose(rng).unwrap().from_existing();
            parent.mutate(config, rng, innovation);
            parent
        };

        child.mutate(config, rng, innovation);
        child
    }

    pub fn cull(&mut self) -> usize {
        // Sort genomes so that the best fitness is at the end.
        self.genomes.sort_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Keep only the top 50% of genomes
        let survivors = (self.genomes.len() as f32 / 2.0).ceil() as usize;
        self.genomes = self.genomes.split_off(self.genomes.len() - survivors);
        survivors
    }
}
