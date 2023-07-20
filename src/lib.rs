use crate::activation::Activation;
use crate::innovation_record::InnovationRecord;

mod activation;
mod connection;
mod genome;
mod node;
mod network;
mod innovation_record;
mod individual;

pub fn init() {
    println!("Hello, world!");

    let innovation_record = &mut InnovationRecord::new(3);
    let mut genome = genome::Genome::new(2, 2, 1, innovation_record);
    genome.output_graph();

    let output = genome.output(&[5.0, 2.0], Activation::Sigmoid);

    genome.output_graph();
    println!("Output: {:?}", output);
}
