use rand::{rngs::StdRng, seq::IndexedRandom, Rng, SeedableRng};

use crate::{
    context::{Environment, NeatConfig},
    genome::genome::Genome,
    species::Species,
    state::{InnovationRecord, SpeciationManager},
};

// Population of the NEAT algorithm
#[derive(Debug)]
pub struct Population {
    // Species manager manages species id and changing compatibility threshold
    pub species_manager: SpeciationManager,
    // Vector of species in the population
    pub species: Vec<Species>,

    // Generation of the population
    pub generation: usize,

    // Configuration and environment for the problem
    pub config: NeatConfig,
    pub environment: Environment,

    // Best genome of this population and its fitness
    pub best_genome: Option<Genome>,
    pub best_fitness: f32,

    // Innovation record for tracking innovation numbers
    pub innovation: InnovationRecord,

    // Shared Rng
    pub rng: StdRng,
}

impl Population {
    // Create empty population
    pub fn new(config: NeatConfig, environment: Environment) -> Self {
        Population {
            species_manager: SpeciationManager::new(
                config.initial_compatibility_threshold,
                0,
                config.target_species_count,
            ),
            species: Vec::new(),
            generation: 0,
            config,
            environment,
            best_genome: None,
            best_fitness: 0.0,
            innovation: InnovationRecord::new(),
            rng: StdRng::from_rng(&mut rand::rng()),
        }
    }

    // Builder to add a seed to the rng
    pub fn with_rng(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    // Builder to initialize population
    pub fn initialize(mut self) -> Self {
        let initial_genome = Genome::create_initial_genome(
            self.environment.input_size,
            self.environment.output_size,
            &self.config,
            &mut self.rng,
            &mut self.innovation,
        );

        // Start with just one species containing all genomes
        let mut initial_species =
            Species::new(self.species_manager.new_species(), initial_genome.clone());

        // Add all genomes to this species, with some diversity
        for _ in 0..self.config.population_size {
            let mut genome = initial_genome.clone();
            // Apply some random mutations to each genome
            for _ in 0..self.rng.random_range(0..=2) {
                genome.mutate(&self.config, &mut self.rng, &mut self.innovation);
            }
            initial_species.genomes.push(genome);
        }

        self.species.push(initial_species);
        self
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
                    < self.species_manager.compatibility_threshold
                {
                    species.genomes.push(genome.clone());
                    placed = true;
                    break;
                }
            }

            // If no suitable species found, create a new one
            if !placed {
                let mut new_species =
                    Species::new(self.species_manager.new_species(), genome.clone());
                new_species.genomes.push(genome.clone());
                self.species.push(new_species);
            }
        }

        // Final cleanup - remove any species that ended up empty
        self.species.retain(|s| !s.genomes.is_empty());

        // Adjust threshold based on current species count
        self.species_manager.adjust_threshold(self.species.len());
    }

    fn reproduce(&mut self) -> Vec<Genome> {
        let mut new_generation = Vec::with_capacity(self.config.population_size);

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

                let mut child = if self.rng.random::<f32>() < self.config.crossover_rate
                    && breeding_pool.len() >= 2
                {
                    // Crossover between two parents
                    let parent1 = breeding_pool.choose(&mut self.rng).unwrap();
                    let parent2 = breeding_pool.choose(&mut self.rng).unwrap();
                    parent1.crossover(parent2, &mut self.rng)
                } else {
                    // Clone a single parent
                    breeding_pool.choose(&mut self.rng).unwrap().from_existing()
                };

                // Apply mutation
                child.mutate(&self.config, &mut self.rng, &mut self.innovation);

                new_generation.push(child);
            }
        }

        // Fill any remaining slots if we didn't reach population size
        while new_generation.len() < self.config.population_size {
            // Create completely new genomes or clone the best one
            if let Some(ref best) = self.best_genome {
                let mut child = best.from_existing();
                child.mutate(&self.config, &mut self.rng, &mut self.innovation);
                new_generation.push(child);
            } else {
                // No best genome yet, create from initial template
                let mut child = Genome::create_initial_genome(
                    self.environment.input_size,
                    self.environment.output_size,
                    &self.config,
                    &mut self.rng,
                    &mut self.innovation,
                );
                child.mutate(&self.config, &mut self.rng, &mut self.innovation);
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
