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
        info.push_str(&format!("Champion: {}\n", self.champion.as_ref().unwrap().fitness));
        // Global average fitness
        let global_avg_fitness = self
            .genomes
            .iter()
            .fold(0.0, |acc, genome| acc + genome.fitness)
            / self.genomes.len() as f64;
        info.push_str(&format!("Global Average Fitness: {}\n", global_avg_fitness));
        info
    }

    fn find_best_genome(&mut self) {
        // Find champion from genomes
        self.genomes.sort();
        let best_genome = self.genomes.first().unwrap();
        if self.champion.is_none() || best_genome.fitness > self.champion.as_ref().unwrap().fitness
        {
            self.champion = Some(best_genome.clone());
        }
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

    pub fn evolve(&mut self) {
        self.find_best_genome();
        let previous_best = self.champion.clone().unwrap();

        let global_avg_fitness = self
            .genomes
            .iter()
            .fold(0.0, |acc, genome| acc + genome.fitness)
            / self.genomes.len() as f64;

        let mut children: Vec<Genome> = vec![];
        for specie in &mut self.species {
            specie.cull();
            if specie.genomes.len() == 0 {
                continue;
            }
            let num_children =
                ((specie.calculate_average_fitness() / global_avg_fitness) * self.genomes.len() as f64).floor() as usize;
            for _ in 0..num_children {
                let child = specie.make_child(&mut self.innovation_record);
                children.push(child);
            }
        }
        while children.len() < self.genomes.len() {
            // Mutate a random new genome from the champion
            let mut new_genome = previous_best.clone();
            new_genome.mutate(&mut self.innovation_record);
            children.push(new_genome);
        }


        self.genomes = vec![];
        children.sort();
        children.truncate(self.population_size);
        self.genomes = children.clone();
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
