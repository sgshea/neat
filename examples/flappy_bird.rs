mod helper_functions;

use neat::population;
use macroquad::prelude::*;
use neat::genome::Genome;
use helper_functions::flappy::*;

async fn eval_genomes(genomes: &mut Vec<Genome>, display: bool) {
    let mut game = Game::new(genomes.len());
    genomes.sort();
    loop {
        let over = game.update();
        if over {
            break;
        }
        for i in 0..genomes.len() {
            let (flappy, pipe1, pipe2, x_dist, velocity) = game.individual_inputs(i);
            let inputs = vec![flappy, pipe1, pipe2, x_dist, velocity];
            let output = genomes[i].feed_forward(inputs.clone());
            if output[0] > 0.55 {
                game.flappy[i].jump();
            }
            if display {
                let text = format!("1_dist: {:?}", inputs[1]);
                draw_text(&text, 410.0, 40.0, 30.0, BLACK);
                let text = format!("2_dist: {:?}", inputs[2]);
                draw_text(&text, 410.0, 60.0, 30.0, BLACK);
                let text = format!("outputs: {:?}", output);
                draw_text(&text, 10.0, 60.0, 30.0, BLACK);
            }
        }
        game.draw();
        next_frame().await
    }
    for i in 0..genomes.len() {
        genomes[i].fitness = game.flappy[i].score;
    }
}

#[macroquad::main("BasicShapes")]
async fn main() {

    let mut population = population::Population::new(350, 5, 1, 0);

    for _ in 0..40 {
        eval_genomes(&mut population.genomes, false).await;
        population.evolve();
        println!("{}", population.get_info());
    }

    if let Some(ref champion) = population.champion {
        loop {
            eval_genomes(vec![champion.clone()].as_mut(), true).await;
        }
    }
    println!("{}", population.get_info());

}