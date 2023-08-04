use crate::environment::Environment;
use crate::individual::Individual;
use crate::innovation_record::InnovationRecord;
use rand::Rng;

#[derive(Debug)]
pub struct Specie {
    id: usize,
    age: usize,
    average_fitness: Option<f64>,
    best_fitness: Option<f64>,
    previous_fitness: Option<f64>,
    representative: Individual,
    individuals: Vec<Individual>,
}

impl Specie {
    pub fn new(id: usize, representative: Individual) -> Self {
        Self {
            id,
            age: 0,
            average_fitness: Option::from(representative.fitness),
            best_fitness: Option::from(representative.fitness),
            previous_fitness: None,
            representative,
            individuals: Vec::new(),
        }
    }

    pub fn set_individuals(&mut self, individuals: Vec<Individual>) {
        self.individuals = individuals;
    }

    // Test new individual against representative to see if it is close enough to add to species
    pub fn add_to_species(&mut self, threshold: f64, individual: Individual) -> bool {
        let distance = self.representative.distance(&individual);
        if distance <= threshold {
            self.individuals.push(individual);
            true
        } else {
            false
        }
    }

    // find best individual
    pub fn find_champion(&self) -> Individual {
        let mut best = &self.individuals[0];
        // Find individual with highest fitness
        for individual in &self.individuals {
            if individual.fitness > best.fitness {
                best = individual;
            }
        }

        best.clone()
    }

    fn make_child(
        &self,
        individual: Individual,
        population: &Vec<Individual>,
        innovation_record: &mut InnovationRecord,
    ) -> Individual {
        if rand::random::<f64>() < 0.25 || population.len() < 2 {
            individual.mutate(innovation_record)
        } else {
            let mut rng = rand::thread_rng();
            // get random individual from population to mate with
            let other = &population[rng.gen_range(0..population.len())];
            individual.crossover(&other)
        }
    }

    // Create next generation of species by mating current individuals
    fn generate_offspring(&mut self, innovation_record: &mut InnovationRecord) -> Vec<Individual> {
        let num_individuals = self.individuals.len();

        let mut individuals_to_mate = (self.individuals.len() as f64 * 1f64) as usize;
        if individuals_to_mate < 1 {
            individuals_to_mate = 1;
        }

        self.individuals.sort();
        self.individuals.truncate(individuals_to_mate);

        let mut rng = rand::thread_rng();

        let mut offspring = Vec::new();
        let champion = self.find_champion();
        offspring.push(champion.clone());

        let mut selected = vec![];
        for _ in 0..num_individuals - 1 {
            selected.push(rng.gen_range(0..self.individuals.len()));
        }

        for x in selected {
            offspring.push(self.make_child(
                self.individuals[x].clone(),
                &self.individuals,
                innovation_record,
            ));
        }

        offspring
    }

    // Mutate all individuals in species
    fn mutate_all(&mut self, innovation_record: &mut InnovationRecord) {
        for individual in &mut self.individuals {
            individual.mutate_in_place(innovation_record);
        }
    }

    // This is the function to be called when evolving
    pub fn update_species(
        &mut self,
        innovation_record: &mut InnovationRecord,
        environment: &mut dyn Environment,
    ) {
        self.age += 1;
        self.previous_fitness = self.best_fitness;

        // Activate all individuals
        for individual in &mut self.individuals {
            environment.evaluate(individual);
        }

        // Get champion fitness
        let best = self.find_champion();
        self.best_fitness = Option::from(best.fitness);
        // Update champion
        self.representative = best;

        // Update average fitness
        let total_fitness = self.individuals.iter().fold(0.0, |acc, x| acc + x.fitness);
        self.average_fitness = Option::from(total_fitness / self.individuals.len() as f64);

        println!(
            "Species {} - Population {} - Age: {} - Best: {} - Average: {}",
            self.id,
            self.individuals.len(),
            self.age,
            self.best_fitness.unwrap(),
            self.average_fitness.unwrap()
        );

        // Generate offspring
        let offspring = self.generate_offspring(innovation_record);
        self.individuals = offspring;
    }
}
