use neat::{
    context::{Environment, NeatConfig},
    genome::genome::Genome,
    nn::{
        feedforward::FeedforwardNetwork,
        nn::{NetworkType, NeuralNetwork},
    },
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
    let config = NeatConfig::builder()
        .network_type(NetworkType::Feedforward)
        .complexity_control(0.005, 0.0015, 1, 3)
        .build();

    let environment = Environment::new(2, 1);
    let mut population = Population::new(config, environment)
        .with_rng(42)
        .initialize();

    for _ in 0..250 {
        population.evaluate_parallel(|genome| xor_test(&genome, false));
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
