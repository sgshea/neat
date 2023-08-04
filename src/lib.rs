use crate::environment::Environment;
use crate::individual::Individual;

mod activation;
mod connection;
mod environment;
mod genome;
mod individual;
mod innovation_record;
mod neat;
mod node;
mod specie;

struct XOR;
impl Environment for XOR {
    fn evaluate(&mut self, individual: &mut Individual) {
        let inputs = vec![
            vec![0f32, 0f32],
            vec![0f32, 1f32],
            vec![1f32, 0f32],
            vec![1f32, 1f32],
        ];

        let expected_outputs = vec![vec![0f32], vec![1f32], vec![1f32], vec![0f32]];

        let mut error = 0.0;
        for i in 0..inputs.len() {
            let output = individual.activate(inputs[i].clone());
            error += output[0] - expected_outputs[i][0];
        }

        // let fitness = 16f32 / (1f32 + error);
        let fitness = 4.0 - error;
        individual.fitness = fitness as f64;
    }
}

fn xor() {
    let config = neat::NeatConfig::new();
    let mut neat = neat::Neat::new(config, 50, 2, 1, 0);

    let mut environment = XOR;
    let mut champion: Option<Individual> = None;

    for _ in 0..50 {
        let test_champ = neat.evaluate(&mut environment);
        if test_champ.clone().is_some() {
            champion = test_champ;
        }
    }

    // For the final XOR test, inputs are 1, 1 and the output node should be as close to 0 as possible

    if let Some(champion) = champion {
        println!("Champion: {:?}", champion);
        champion.output_graph();
    }
}

pub fn init() {
    xor();
}
