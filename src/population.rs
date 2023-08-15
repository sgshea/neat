use macroquad::rand::ChooseRandom;
use crate::genome::Genome;
use crate::innovation_record::InnovationRecord;
use crate::species::Specie;

pub struct Population {
    pub genomes: Vec<Genome>,
    species: Vec<Specie>,

    // Includes bias node
    pub input_num: usize,

    pub output_num: usize,
    pub hidden_num: usize,
    pub population_size: usize,

    pub age: usize,
    pub champion: Option<Genome>,

    innovation_record: InnovationRecord,
}

impl Population {
    pub fn new(population_size: usize, inputs: usize, outputs: usize, hidden: usize) -> Self {
        let mut population = Self {
            genomes: vec![],
            species: vec![],
            input_num: inputs,
            output_num: outputs,
            hidden_num: hidden,
            population_size,
            age: 0,
            champion: None,
            innovation_record: InnovationRecord::new(),
        };

        let genome = Genome::new(inputs, outputs, &mut population.innovation_record);
        for _ in 0..population_size {
            let mut new_genome = genome.clone();
            new_genome.mutate(&mut population.innovation_record);
            population.genomes.push(new_genome);
        }

        population
    }

    pub fn get_info(&self) -> String {
        let mut info = String::new();
        info.push_str(&format!("Population Size: {}\n", self.population_size));
        info.push_str(&format!("Species: {}\n", self.species.len()));
        info.push_str(&format!("Age: {}\n", self.age));
        let champion = self.champion.as_ref().unwrap();
        info.push_str(&format!("Champion: {}, Adjusted: {}, Nodes: {}, Genes: {}\n",
                               champion.fitness, champion.adj_fitness, champion.node.len(), champion.genes.len()));
        // Global average fitness
        let global_avg_fitness = self
            .genomes
            .iter()
            .fold(0.0, |acc, genome| acc + genome.fitness)
            / self.genomes.len() as f64;
        info.push_str(&format!("Global Average Fitness: {}\n", global_avg_fitness));
        info
    }

    fn speciate(&mut self) {
        // Remove empty species
        self.species.retain(|specie| !specie.genomes.is_empty());

        for specie in &mut self.species {
            specie.representative = specie.select_genome();
            specie.genomes = vec![];
        }

        for genome in &mut self.genomes {
            let mut found_specie = false;
            'inner: for specie in &mut self.species {
                if specie.match_genome(genome) {
                    specie.add_genome(genome.clone());
                    found_specie = true;
                    break 'inner;
                }
            }
            if !found_specie {
                let new_specie = Specie::new(self.species.len(), genome.clone());
                self.species.push(new_specie);
            }
        }

        // Remove empty species
        self.species.retain(|specie| !specie.genomes.is_empty());
    }

    fn generate_generation(&mut self) -> Vec<Genome> {
        // Adjust fitness
        let mut total_adjusted_fitness = 0.0;
        for specie in &mut self.species {
            total_adjusted_fitness += specie.calculate_average_fitness();
        }
        total_adjusted_fitness /= self.population_size as f64;

        // Generate new generation
        let mut new_genomes = vec![];
        for specie in &mut self.species {
            if specie.stagnation > 15 || specie.genomes.is_empty() {
                continue;
            }
            let specie_size = specie.cull();
            // dbg!(specie_size);
            // dbg!(specie.average_fitness);
            let mut offspring_num = ((specie.average_fitness / total_adjusted_fitness) * specie_size as f64) as usize;
            // dbg!(offspring_num);
            if offspring_num < 1 {
                offspring_num = 1;
            }
            for _ in 0..offspring_num {
                let new_genome = specie.make_child(&mut self.innovation_record);
                new_genomes.push(new_genome);
            }
        }

        // Add new genomes to fill up population
        while new_genomes.len() < self.population_size {
            let mut genome = self.genomes.choose().unwrap().clone();
            genome.mutate(&mut self.innovation_record);
            new_genomes.push(genome);
        }

        new_genomes
    }

    pub fn evolve(&mut self) {
        // Get new champion
        self.genomes.sort();
        let champion = self.genomes[0].clone();
        if self.champion.is_none() || champion.fitness > self.champion.as_ref().unwrap().fitness {
            self.champion = Some(champion.clone());
        }

        // Generate new generation
        let mut new_genomes = self.generate_generation();
        // Add champion to new generation
        new_genomes.push(champion);
        self.genomes = new_genomes;
        self.speciate();
        self.age += 1;
    }

    pub fn evaluate(&mut self, f: &dyn Fn(&mut Genome, bool)) {
        for genome in &mut self.genomes {
            f(genome, false);
        }
        self.evolve();
    }

    pub fn evaluate_whole(&mut self, f: &dyn Fn(&mut Vec<Genome>, bool)) {
        f(&mut self.genomes, false);
        self.evolve();
    }
}
