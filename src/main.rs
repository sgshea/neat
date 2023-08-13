use crate::genome::Genome;

mod genes;
mod genome;
mod innovation_record;
mod population;
mod species;

fn eval_genomes(genome: &mut Genome, display: bool) {
    let xor = vec![
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
    ];

    let mut fitness = 0.0;
    for (xi, xo) in &xor {
        let output = genome.feed_forward(xi.clone());
        fitness += (xo[0] - output[0]).powi(2);
        if display {
            println!("input: {:?}", xi);
            println!("output: {:?}", output);
            println!("error: {}\n\n", (output[0] - xo[0]).powf(2.0));
        }
    }
    genome.fitness = 4.0 - fitness;
}

fn main() {

    for _ in 0..10 {
        let mut population = population::Population::new(150, 2, 1, 0);
        for _ in 0..35 {
            population.evaluate(&eval_genomes);
        }
        if let Some(ref champion) = population.champion {
            eval_genomes(&mut champion.clone(), false);
        }
        println!("{}", population.get_info());
    }
}
