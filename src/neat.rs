use crate::activation::Activation;
use crate::environment::Environment;
use crate::individual::Individual;
use crate::innovation_record::InnovationRecord;
use crate::specie::Specie;

#[derive(Debug)]
pub struct NeatConfig {
    pub compatibility_threshold: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
    pub delta_t: f64,
    pub delta_disjoint: f64,
    pub delta_weights: f64,
    pub delta_threshold: f64,
    pub weight_mutation_chance: f64,
    pub weight_mutation_power: f64,
    pub node_mutation_chance: f64,
    pub connection_mutation_chance: f64,
    pub bias_mutation_chance: f64,
    pub enable_mutation_chance: f64,
    pub disable_mutation_chance: f64,
    pub crossover_chance: f64,
    pub crossover_mate_chance: f64,
    pub activation: Activation,
}

impl NeatConfig {
    // Construct with default values
    pub fn new() -> Self {
        Self {
            compatibility_threshold: 3.0,
            c1: 1.0,
            c2: 1.0,
            c3: 0.4,
            delta_t: 1.0,
            delta_disjoint: 2.0,
            delta_weights: 0.4,
            delta_threshold: 1.0,
            weight_mutation_chance: 0.8,
            weight_mutation_power: 0.5,
            node_mutation_chance: 0.03,
            connection_mutation_chance: 0.05,
            bias_mutation_chance: 0.08,
            enable_mutation_chance: 0.2,
            disable_mutation_chance: 0.2,
            crossover_chance: 0.75,
            crossover_mate_chance: 0.5,
            activation: Activation::Sigmoid,
        }
    }
}

#[derive(Debug)]
pub struct Neat {
    config: NeatConfig,
    pub population_size: usize,
    pub input_nodes: usize,
    pub output_nodes: usize,
    pub hidden_nodes: usize,

    innovation_record: InnovationRecord,

    pub species: Vec<Specie>,
    champion_fitness: Option<f64>,

    // Champion is none until reaching a specified fitness threshold
    pub champion: Option<Individual>,
}

impl Neat {

    // Creates a new NEAT population
    pub fn new(
        config: NeatConfig,
        population_size: usize,
        input_nodes: usize,
        output_nodes: usize,
        hidden_nodes: usize,
    ) -> Self {

        let mut neat = Neat {
            config,
            population_size,
            input_nodes,
            output_nodes,
            hidden_nodes,
            innovation_record: InnovationRecord::new(input_nodes + output_nodes + hidden_nodes),
            species: vec![],
            champion_fitness: None,
            champion: None,
        };

        neat.init_population(population_size, input_nodes, output_nodes, hidden_nodes);

        neat
    }

    // Initialize the population, creating individuals and adding to a single species
    fn init_population(
        &mut self,
        population_size: usize,
        input_nodes: usize,
        output_nodes: usize,
        hidden_nodes: usize,
    ) {
        let mut individuals = vec![];

        for _ in 0..population_size {
            individuals.push(Individual::new(
                input_nodes,
                output_nodes,
                hidden_nodes,
                &mut self.innovation_record,
            ));
        }

        let mut specie = Specie::new(self.innovation_record.new_species(), individuals.first().unwrap().clone());
        specie.set_individuals(individuals);
        self.species.push(specie);
    }

    // Eval to next generation
    fn next_generation(&mut self, environment: &mut dyn Environment) {
        // Update each species
        for specie in &mut self.species {
            specie.update_species(&mut self.innovation_record, environment);
        }
    }

    pub fn has_champion(&self) -> bool {
        self.champion.is_some()
    }

    // Evaluate the current population
    // Returns the champion if one
    pub fn evaluate(&mut self, environment: &mut dyn Environment) -> Option<Individual> {

        // Evolve next generation
        self.next_generation(environment);

        // Find global champion
        // TODO: only one species right now
        let champion = self.species.first().unwrap().find_champion();

        self.champion = Some(champion.clone());
        self.champion_fitness = Some(champion.fitness);

        self.champion.clone()

    }
}
