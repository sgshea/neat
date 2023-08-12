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
        if self.champion.is_none() || best_genome.fitness > self.champion.as_ref().unwrap().fitness {
            self.champion = Some(best_genome.clone());
        }
    }

    fn cull_species(&mut self) {
        for specie in &mut self.species {
            specie.cull();
            specie.fitness_sharing();
            specie.set_average();
        }
    }

    fn speciate(&mut self) {
        for specie in &mut self.species {
            specie.genomes.clear();
        }

        for genome in &mut self.genomes {
            let mut found_specie = false;
            for specie in &mut self.species {
                if specie.match_genome(genome) {
                    specie.add_genome(genome.clone());
                    found_specie = true;
                    break;
                }
            }
            if !found_specie {
                let new_specie = Specie::new(self.species.len(), genome.clone());
                self.species.push(new_specie);
            }
        }
    }

    pub fn evolve(&mut self) {
        self.find_best_genome();
        let previous_best = self.champion.clone().unwrap();

        self.cull_species();

        let average_sum = self
            .species
            .iter()
            .map(|specie| specie.average_fitness)
            .sum::<f64>();
        let mut children: Vec<Genome> = vec![];
        for specie in &mut self.species {
            let num_children =
                (specie.average_fitness / average_sum * self.genomes.len() as f64).floor() as usize;
            for _ in 0..num_children {
                let child = specie.make_child(&mut self.innovation_record);
                children.push(child);
            }
        }
        while children.len() < self.genomes.len() {
            children.push(previous_best.clone());
        }

        self.genomes.clear();
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
}
