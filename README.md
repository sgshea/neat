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

The library should be interacted through the `Population` struct. Construct a population and use the
`evaluate` function to evaluate the population. The evaluate function
should run the experiment and assign fitness values to the genomes.

### Example

- XOR function approximation at [examples/xor.rs](examples/xor.rs)
