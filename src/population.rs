use rand::{seq::IndexedRandom, Rng};

use crate::{
    environment::Environment,
    genome::genome::{Genome, InnovationRecord},
    species::Species,
};

#[derive(Debug, Clone)]
pub struct NeatConfig {
    // General parameters
    pub population_size: usize,

    // Compatibility parameters
    pub compatibility_threshold: f32,
    pub compatibility_disjoint_coefficient: f32,
    pub compatibility_weight_coefficient: f32,

    // Mutation parameters
    pub weight_mutation_prob: f32,
    pub weight_perturb_prob: f32,
    pub new_connection_prob: f32,
    pub new_node_prob: f32,
    pub toggle_enable_prob: f32,

    // Reproduction parameters
    pub crossover_rate: f32,
    pub survival_threshold: f32,

    // Speciation parameters
    pub species_elitism: bool,
    pub elitism: usize,
    pub stagnation_limit: usize,
    pub target_species_count: usize,
}

impl NeatConfig {
    pub fn default() -> Self {
        NeatConfig {
            population_size: 150,

            compatibility_threshold: 3.0,
            compatibility_disjoint_coefficient: 1.0,
            compatibility_weight_coefficient: 0.3,

            weight_mutation_prob: 0.8,
            weight_perturb_prob: 0.9,
            new_connection_prob: 0.05,
            new_node_prob: 0.03,
            toggle_enable_prob: 0.01,

            crossover_rate: 0.75,
            survival_threshold: 0.2,

            species_elitism: true,
            elitism: 1,
            stagnation_limit: 35,
            target_species_count: 15,
        }
    }
}

#[derive(Debug)]
pub struct Population {
    pub species: Vec<Species>,
    pub species_counter: usize,
    pub generation: usize,
    pub config: NeatConfig,
    pub environment: Environment,
    pub best_genome: Option<Genome>,
    pub best_fitness: f32,
    pub innovation: InnovationRecord,

    pub initial_genome: Genome,
}

impl Population {
    pub fn new(config: NeatConfig, environment: Environment) -> Self {
        let mut innovation = InnovationRecord::new();
        let initial_genome = Genome::create_initial_genome(
            environment.input_size,
            environment.output_size,
            &mut innovation,
        );

        let mut population = Population {
            species: Vec::new(),
            species_counter: 0,
            generation: 0,
            config,
            environment,
            best_genome: None,
            best_fitness: 0.0,
            innovation,
            initial_genome,
        };

        // Start with just one species containing all genomes
        let mut initial_species = Species::new(
            population.species_counter,
            population.initial_genome.clone(),
        );
        population.species_counter += 1;

        // Add all genomes to this species, with some diversity
        for _ in 0..population.config.population_size {
            let mut genome = population.initial_genome.clone();
            // Apply some random mutations to each genome
            for _ in 0..rand::rng().random_range(0..=2) {
                genome.mutate(&population.config, &mut population.innovation);
            }
            initial_species.genomes.push(genome);
        }

        population.species.push(initial_species);
        population
    }

    pub fn evaluate<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&Genome) -> f32,
    {
        for species in &mut self.species {
            for genome in &mut species.genomes {
                genome.fitness = fitness_fn(genome);
            }
        }
    }

    pub fn get_best_genome(&self) -> Option<&Genome> {
        self.best_genome.as_ref()
    }

    fn speciate(&mut self, new_generation: &Vec<Genome>) {
        // Clear current species
        for specie in &mut self.species {
            specie.genomes.clear();
        }

        // Keep track of empty species to remove later
        let mut empty_species = Vec::new();

        // Update representatives for existing species
        for species in &mut self.species {
            if !species.genomes.is_empty() {
                species.representative = species.select_representative();
            } else {
                empty_species.push(species.id);
            }
        }

        // Remove empty species
        self.species.retain(|s| !empty_species.contains(&s.id));

        // Assign each genome to a species
        for genome in new_generation {
            let mut placed = false;

            // Try to find an existing species
            for species in &mut self.species {
                if species
                    .representative
                    .compatibility_distance(genome, &self.config)
                    < self.config.compatibility_threshold
                {
                    species.genomes.push(genome.clone());
                    placed = true;
                    break;
                }
            }

            // If no suitable species found, create a new one
            if !placed {
                let mut new_species = Species::new(self.species_counter, genome.clone());
                new_species.genomes.push(genome.clone());
                self.species.push(new_species);
                self.species_counter += 1;
            }
        }

        // Final cleanup - remove any species that ended up empty
        self.species.retain(|s| !s.genomes.is_empty());

        // If we have too many species, increase threshold slightly
        if self.species.len() > self.config.target_species_count * 2 {
            self.config.compatibility_threshold *= 1.05;
        }
        // If we have too few species, decrease threshold slightly
        else if self.species.len() < self.config.target_species_count / 2
            && self.species.len() > 1
        {
            self.config.compatibility_threshold *= 0.95;
        }
    }

    fn reproduce(&mut self) -> Vec<Genome> {
        let mut new_generation = Vec::with_capacity(self.config.population_size);
        let mut rng = rand::rng();

        // Step 1: Update species statistics and adjust fitness
        let mut total_adjusted_fitness = 0.0;
        let mut stagnant_species = Vec::new();

        // Calculate stats for each species
        for species in &mut self.species {
            species.calculate_average_fitness();
            let amount = species.genomes.len();

            // Update best fitness and check for stagnation
            let _ = species.update_best_fitness();

            // Update global best genome if necessary
            if let Some(ref best) = species.best_fitness_genome {
                if best.fitness > self.best_fitness {
                    self.best_fitness = best.fitness;
                    self.best_genome = Some(best.clone());
                }
            }

            // Mark stagnant species for potential removal
            if species.staleness >= self.config.stagnation_limit {
                stagnant_species.push(species.id);
            }

            // Calculate adjusted fitness
            for genome in &mut species.genomes {
                genome.adjusted_fitness = genome.fitness / amount as f32;
                total_adjusted_fitness += genome.adjusted_fitness;
            }
        }

        // Remove stagnant species, but keep at least one
        if self.species.len() > 1 {
            self.species.retain(|s| !stagnant_species.contains(&s.id));
        }

        // Step 2: Determine number of offspring for each species
        let mut offspring_per_species = Vec::new();

        for species in &self.species {
            // Sum of adjusted fitness in this species
            let species_adjusted_fitness: f32 =
                species.genomes.iter().map(|g| g.adjusted_fitness).sum();

            // Calculate proportion of total population this species should produce
            let offspring_count = if total_adjusted_fitness > 0.0 {
                (species_adjusted_fitness / total_adjusted_fitness
                    * self.config.population_size as f32)
                    .round() as usize
            } else {
                // If all fitness is 0, distribute evenly
                self.config.population_size / self.species.len()
            };

            offspring_per_species.push(offspring_count);
        }

        // Step 3: Elitism - preserve the best genomes directly
        if self.config.elitism > 0 {
            for species in &self.species {
                // If this species has enough members for elitism
                if species.genomes.len() >= self.config.elitism {
                    // Sort by fitness (highest last)
                    let mut genomes = species.genomes.clone();
                    genomes.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

                    // Take the N best genomes directly
                    for i in 0..self.config.elitism.min(genomes.len()) {
                        if new_generation.len() < self.config.population_size {
                            new_generation.push(genomes[genomes.len() - 1 - i].clone());
                        }
                    }
                }
            }
        }

        // Step 4: Create offspring through crossover and mutation
        for (species_idx, &offspring_count) in offspring_per_species.iter().enumerate() {
            let species = &self.species[species_idx];

            // Skip if this species should have no offspring
            if offspring_count == 0 || species.genomes.is_empty() {
                continue;
            }

            // Cull the species first (keep only top percentage)
            let mut breeding_pool = species.genomes.clone();
            breeding_pool.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
            let cutoff =
                (breeding_pool.len() as f32 * self.config.survival_threshold).ceil() as usize;
            if cutoff > 0 && cutoff < breeding_pool.len() {
                breeding_pool = breeding_pool.split_off(breeding_pool.len() - cutoff);
            }

            // Create offspring
            for _ in 0..offspring_count {
                if new_generation.len() >= self.config.population_size {
                    break;
                }

                if breeding_pool.is_empty() {
                    continue;
                }

                let mut child = if rng.random::<f32>() < self.config.crossover_rate
                    && breeding_pool.len() >= 2
                {
                    // Crossover between two parents
                    let parent1 = breeding_pool.choose(&mut rng).unwrap();
                    let parent2 = breeding_pool.choose(&mut rng).unwrap();
                    parent1.crossover(parent2)
                } else {
                    // Clone a single parent
                    breeding_pool.choose(&mut rng).unwrap().from_existing()
                };

                // Apply mutation
                child.mutate(&self.config, &mut self.innovation);

                new_generation.push(child);
            }
        }

        // Fill any remaining slots if we didn't reach population size
        while new_generation.len() < self.config.population_size {
            // Create completely new genomes or clone the best one
            if let Some(ref best) = self.best_genome {
                let mut child = best.from_existing();
                child.mutate(&self.config, &mut self.innovation);
                new_generation.push(child);
            } else {
                // No best genome yet, create from initial template
                let mut child = self.initial_genome.from_existing();
                child.mutate(&self.config, &mut self.innovation);
                new_generation.push(child);
            }
        }

        new_generation
    }

    pub fn evolve(&mut self) {
        // Increment generation counter.
        self.generation += 1;

        // Step 2: Produce a new generation.
        let new_generation = self.reproduce();

        // Step 3: Regroup genomes into species.
        self.speciate(&new_generation);
    }
}
