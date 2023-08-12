use crate::genome::Genome;
use crate::innovation_record::InnovationRecord;
use crate::species::Specie;

pub struct Population {
    genomes: Vec<Genome>,
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

    fn find_best_genome(&mut self) {
        // Find champion from genomes
        self.genomes.sort();
        let best_genome = self.genomes.last().unwrap().clone();
        if self.champion.is_none() || best_genome.fitness > self.champion.as_ref().unwrap().fitness
        {
            self.champion = Some(best_genome.clone());
        }
    }

    fn speciate(&mut self) {
        for specie in &mut self.species {
            specie.representative = specie.select_genome();
            specie.genomes.clear();
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

        // Adjust fitness for species
        for specie in &mut self.species {
            specie.fitness_sharing();
        }

        let global_avg_fitness = self
            .genomes
            .iter()
            .fold(0.0, |acc, genome| acc + genome.fitness)
            / self.genomes.len() as f64;

        let mut children: Vec<Genome> = vec![];
        for specie in &mut self.species {
            let num_children =
                ((specie.calculate_average_fitness() / global_avg_fitness) * specie.genomes.len() as f64).floor() as usize;
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

        self.genomes.clear();
        self.genomes = children;
        self.speciate();
        self.age += 1;

        //
        println!("Species amount: {}", self.species.len());
    }

    pub fn evaluate(&mut self, f: &dyn Fn(&mut Genome, bool)) {
        for genome in &mut self.genomes {
            f(genome, false);
        }
        self.evolve();
    }
}
