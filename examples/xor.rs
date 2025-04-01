use neat::{
    context::{Environment, NeatConfig},
    genome::genome::Genome,
    nn::{feedforward::FeedforwardNetwork, nn::NeuralNetwork},
    population::Population,
};

fn xor_test(genome: &Genome, display: bool) -> f32 {
    let xor = vec![
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
    ];

    let mut nn = FeedforwardNetwork::new(genome).unwrap();

    let mut fitness = 4.0;

    for (xi, xo) in &xor {
        let output = nn.activate(&xi.clone()).unwrap();
        fitness -= (xo - output[0]).powf(2.0);
        if display {
            println!("input: {:?}, output: {:?}, expected: {:?}", xi, output, xo);
        }
    }
    fitness
}

fn main() {
    let config = NeatConfig {
        population_size: 150,

        initial_compatibility_threshold: 3.0,
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
        target_species_count: 10,
    };
    let environment = Environment::new(2, 1);
    let mut population = Population::new(config, environment)
        .with_rng(42)
        .initialize();

    for _ in 0..100 {
        population.evaluate(|genome| xor_test(&genome, false));
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

        let fitness = xor_test(best, true);
        println!("Fitness: {}", fitness);
    }
}
