use crate::genome::Genome;
use crate::innovation_record::InnovationRecord;
use rand::seq::SliceRandom;
use rand::Rng;

pub struct Specie {
    pub id: usize,
    pub genomes: Vec<Genome>,
    pub champion: Genome,
    pub representative: Genome,
    pub average_fitness: f64,
    pub staleness: usize,
}

impl Specie {
    pub fn new(id: usize, representative: Genome) -> Self {
        let average_fitness = representative.fitness;

        Self {
            id,
            genomes: vec![representative.clone()],
            champion: representative.clone(),
            representative,
            average_fitness,
            staleness: 0,
        }
    }

    // Does genome fit in species
    pub fn match_genome(&self, genome: &Genome) -> bool {
        self.representative.compatability_distance(genome) < 4.0
    }

    pub fn add_genome(&mut self, genome: Genome) {
        self.genomes.push(genome);
    }

    pub fn set_average(&mut self) {
        let mut total = 0.0;
        for genome in &self.genomes {
            total += genome.fitness;
        }

        self.average_fitness = total / self.genomes.len() as f64;
    }

    pub fn select_genome(&self) -> Genome {
        let mut rng = rand::thread_rng();

        // try and choose a random genome above average fitness
        for _ in 0..self.genomes.len() {
            let genome = self.genomes.choose(&mut rng).unwrap();
            if genome.fitness > self.average_fitness {
                return genome.clone();
            }
        }

        // if none was found just use champion
        self.champion.clone()
    }

    pub fn make_child(&self, innovation_record: &mut InnovationRecord) -> Genome {
        let mut rng = rand::thread_rng();
        let mut child = if rng.gen::<f64>() < 0.25 {
            self.select_genome()
        } else {
            let mut parent_1 = self.select_genome();
            let mut parent_2 = self.select_genome();

            if parent_1 > parent_2 {
                parent_1.crossover(parent_2)
            } else {
                parent_2.crossover(parent_1)
            }
        };
        child.mutate(innovation_record);
        child
    }

    pub fn cull(&mut self) {
        self.genomes.sort();
        // Remove first half (lowest fitness)
        self.genomes.drain(0..self.genomes.len() / 2);
    }

    pub fn fitness_sharing(&mut self) {
        let length = self.genomes.len() as f64;
        for genome in &mut self.genomes {
            genome.fitness /= length;
        }
    }
}
