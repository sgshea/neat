//! Neural Network module.
//!

use crate::genome::genome::Genome;

/// A generic trait for neural networks that can be used with the NEAT library
/// The lifetime parameter 'n represents the lifetime of the genome reference.
pub trait NeuralNetwork<'n> {
    /// Create a neural network by borrowing the genome
    /// This can error if the genome is invalid for the network type
    fn new(genome: &'n Genome) -> Result<Self, NetworkError>
    where
        Self: Sized;

    /// Activate the network for one step/iteration
    fn activate(&mut self, inputs: &[f32]) -> Result<Vec<f32>, NetworkError>;
}

/// Different types of neural networks
pub enum NetworkType {
    Feedforward,
    // CTRNN,
    // LSTM,
    // GRU,
}

/// Error types
#[derive(thiserror::Error, miette::Diagnostic, Debug)]
pub enum NetworkError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Invalid genome: {0}")]
    InvalidGenome(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Cycle detected in network")]
    CycleDetected(String),
}
