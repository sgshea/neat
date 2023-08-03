use crate::environment::Environment;
use crate::individual::Individual;

mod activation;
mod connection;
mod genome;
mod individual;
mod innovation_record;
mod neat;
mod node;
mod specie;
mod environment;

struct XOR;
impl Environment for XOR {
    fn evaluate(&mut self, individual: &mut Individual) {
        let mut output = vec![0f32];
        let mut distance: f32;
        output = individual.activate(vec![0f32, 0f32]);
        distance = (0f32 - output[0]).powi(2);
        output = individual.activate(vec![0f32, 1f32]);
        distance = (1f32 - output[0]).powi(2);
        output = individual.activate(vec![1f32, 0f32]);
        distance = (1f32 - output[0]).powi(2);
        output = individual.activate(vec![1f32, 1f32]);
        distance = (0f32 - output[0]).powi(2);

        let fitness = 16f32 / (1f32 + distance);

        individual.fitness = fitness as f64;
    }
}

pub fn init() {
    println!("Hello, world!");

    let config = neat::NeatConfig::new();
    let mut neat = neat::Neat::new(config, 150, 2, 1, 0);

    let mut environment = XOR;
    let mut champion: Option<Individual> = None;

    for _ in 0..55 {
        let test_champ = neat.evaluate(&mut environment);
        if test_champ.clone().is_some() {
            champion = test_champ;
        }
    }

    // A fitness of around >15 is considered a success
    // For the final XOR test, inputs are 1, 1 and the output node should be as close to 0 as possible

    if let Some(champion) = champion {
        println!("Champion: {:?}", champion);
        champion.output_graph();
    }

}
