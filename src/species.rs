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
    pub stagnation: usize,
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
            stagnation: 0,
        }
    }

    // Does genome fit in species
    pub fn match_genome(&mut self, genome: &Genome) -> bool {
        self.representative.compatability_distance(genome) < 2.0
    }

    pub fn add_genome(&mut self, genome: Genome) {
        self.genomes.push(genome);
    }

    // Calculates average fitness of species
    // Returns sum of adj fitness
    pub fn calculate_average_fitness(&mut self) -> f64 {
        let genome_count = self.genomes.len() as f64;

        // Fitness sharing
        self.genomes.iter_mut().for_each(|genome| {
            genome.adj_fitness = genome.fitness / genome_count;
        });

        let total = self.genomes.iter().fold(0.0, |acc, genome| acc + genome.adj_fitness);

        let fitness = total / genome_count;

        // Check stagnation
        if fitness > self.average_fitness {
            self.stagnation = 0;
        } else {
            self.stagnation += 1;
        }

        self.average_fitness = fitness;
        total
    }

    pub fn select_genome(&self) -> Genome {
        let mut rng = rand::thread_rng();
        self.genomes.choose(&mut rng).unwrap().clone()
    }

    pub fn make_child(&self, innovation_record: &mut InnovationRecord) -> Genome {
        let mut rng = rand::thread_rng();
        let mut child = if rng.gen::<f64>() < 0.25 {
            let mut parent = self.select_genome();
            parent.mutate(innovation_record);
            parent
        } else {
            let mut parent_1 = self.select_genome();
            let mut parent_2 = self.select_genome();

            if parent_1 < parent_2 {
                parent_1.crossover(parent_2)
            } else {
                parent_2.crossover(parent_1)
            }
        };
        child.mutate(innovation_record);
        child
    }

    pub fn cull(&mut self) -> usize {
        let prev_len = self.genomes.len();
        self.genomes.sort();
        // Remove second half (lowest fitness)
        if prev_len > 3 {
            self.genomes.truncate(self.genomes.len() / 2);
        }
        prev_len
    }
}
