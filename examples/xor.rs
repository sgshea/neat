use neat::genome::Genome;
use neat::population;

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

    let mut best_champion: Option<Genome> = None;
    for _ in 0..45 {
        let mut population = population::Population::new(50, 2, 1, 0);
        for _ in 0..45 {
            population.evaluate(&eval_genomes);
        }
        if let Some(ref champion) = population.champion {
            if best_champion.is_none() || champion.fitness > best_champion.as_ref().unwrap().fitness {
                best_champion = Some(champion.clone());
            }
            eval_genomes(&mut champion.clone(), false);
        }
        println!("{}", population.get_info());
    }

    if best_champion.is_some() {
        let champ = best_champion.as_ref().unwrap();
        println!("Best Champion: {}", champ);
        eval_genomes(&mut champ.clone(), true);
    }
}
