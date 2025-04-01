use neat::{
    context::{ActivationFunction, Environment, NeatConfig},
    genome::genome::Genome,
    nn::{
        ctrnn::CtrnnNetwork,
        nn::{NetworkType, NeuralNetwork},
    },
    population::Population,
};

fn xor_test_ctrnn(genome: &Genome, display: bool) -> f32 {
    let xor = vec![
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
    ];

    // Create CTRNN with a smaller time step for better accuracy
    let mut nn = CtrnnNetwork::new(genome).unwrap();
    let mut fitness = 4.0;

    for (xi, xo) in &xor {
        // Reset the network state between examples
        nn.reset_states();

        // Run the network for multiple time steps to let it stabilize
        for _ in 0..10 {
            nn.activate(&xi).unwrap();
        }

        // Get the final activation
        let output = nn.activate(&xi).unwrap();

        fitness -= (xo - output[0]).powf(2.0);

        if display {
            println!("input: {:?}, output: {:?}, expected: {:?}", xi, output, xo);
        }
    }

    fitness
}

fn main() {
    let config = NeatConfig {
        network_type: NetworkType::CTRNN,
        bias_mutation_prob: 0.3,
        time_constant_mutation_prob: 0.2,
        param_perturb_prob: 0.9,

        population_size: 80,

        initial_compatibility_threshold: 3.0,
        compatibility_disjoint_coefficient: 1.0,
        compatibility_weight_coefficient: 0.3,

        weight_mutation_prob: 0.8,
        weight_perturb_prob: 0.9,
        new_connection_prob: 0.1, // Allow more connections for recurrent networks
        new_node_prob: 0.05,      // Slightly higher for CTRNNs
        toggle_enable_prob: 0.01,

        crossover_rate: 0.75,
        survival_threshold: 0.3,

        species_elitism: true,
        elitism: 2,
        stagnation_limit: 20,
        target_species_count: 5,

        allowed_activation_functions: vec![ActivationFunction::Sigmoid],
        default_activation_function: ActivationFunction::Sigmoid,

        complexity_penalty_coefficient: 0.001,
        connections_penalty_coefficient: 0.0005,
        target_complexity: 5,    // XOR minimal solution has 1 hidden node
        complexity_threshold: 7, // Don't penalize until more than this
    };

    let environment = Environment::new(2, 1);
    let mut population = Population::new(config, environment)
        .with_rng(42)
        .initialize();

    for _ in 0..100 {
        // More generations might be needed for CTRNN to learn
        population.evaluate(|genome| xor_test_ctrnn(genome, false));
        population.evolve();

        println!(
            "Generation {}: Best fitness = {}, Species: {}",
            population.generation,
            population.best_fitness,
            population.species.len()
        );
    }

    if let Some(best) = population.get_best_genome() {
        println!("Best Genome: {:?}", best);
        println!("Genome node count: {}", best.nodes.len());
        println!("Genome connection count: {}", best.connections.len());

        let fitness = xor_test_ctrnn(best, true);
        println!("Fitness: {}", fitness);
    }
}
