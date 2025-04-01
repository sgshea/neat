use neat::{
    environment::Environment,
    genome::genome::Genome,
    nn::{feedforward::FeedforwardNetwork, nn::NeuralNetwork},
    population::{NeatConfig, Population},
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
    let config = NeatConfig::default();
    let environment = Environment::new(2, 1);
    let mut population = Population::new(config, environment);

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
