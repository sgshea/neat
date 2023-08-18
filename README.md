# Rust Neat

Implementation of [Neuroevolution of Augmenting Topologies (NEAT)](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) in Rust.

NEAT is a genetic algorithm for the generation of evolving artificial neural networks. Evolution is accomplished by
altering both weights and structure of the networks.

## State of the project

Core functionality of NEAT is implemented.
- Genomes hold the structure of the network using a connection and node genes.
- Mutation of genomes through mutating weights, adding connections and nodes.
- Speciation using a compatability distance metric as described in the paper.
  - Fitness sharing to encourage speciation.
- Crossover of genomes when generating the next generation.

## Usage

The library should be interacted through the `Population` struct. Construct a population and use either the
`evaluate` or `evaluate_whole` functions to evaluate the population. Both functions take an evaluation function which
should run the experiment and assign fitness values to the genomes. The `evaluate_whole` function will give access to
all the population's genomes while `evaluate` will only give access to a single genome at a time.

### Example

Currently, the XOR example from the paper is fully implemented, see `examples/xor.rs` for the full code.

```rust
// evaluation function to test XOR problem, assigns fitness to genome at end
// `display` boolean used to control when to print output
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

// abbreviated main function from actual example showing how to construct a population and call
// the evaluation function, as well as displaying output of the best genome
fn main() {
      let mut population = population::Population::new(50, 2, 1, 0);
      for _ in 0..45 {
          population.evaluate(&eval_genomes);
      }
      if let Some(ref champion) = population.champion {
          if best_champion.is_none() || champion.fitness > best_champion.as_ref().unwrap().fitness {
              best_champion = Some(champion.clone());
          }
          eval_genomes(&mut champion.clone(), true);
      }
}
```

## Planned functionality

- [ ] Add support for recurrent networks
- [ ] Add support for subtractive mutations (to allow for networks to find the smallest possible solution)
- [ ] Add more types of examples