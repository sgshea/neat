use crate::nn::nn::NetworkType;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFunction {
    Identity,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    Custom(fn(f32) -> f32),
}

impl ActivationFunction {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Identity => x,
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Relu => x.max(0.0),
            ActivationFunction::LeakyRelu => x.max(0.01 * x),
            ActivationFunction::Custom(f) => f(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Environment {
    pub input_size: usize,
    pub output_size: usize,
}

impl Environment {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Environment {
            input_size,
            output_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeatConfig {
    pub network_type: NetworkType,

    // CTRNN Specific
    pub bias_mutation_prob: f32,
    pub time_constant_mutation_prob: f32,
    pub param_perturb_prob: f32,

    // General parameters
    pub population_size: usize,

    // Compatibility parameters
    pub initial_compatibility_threshold: f32,
    pub compatibility_disjoint_coefficient: f32,
    pub compatibility_weight_coefficient: f32,

    // Mutation parameters
    pub weight_mutation_prob: f32,
    pub weight_perturb_prob: f32,
    pub new_connection_prob: f32,
    pub new_node_prob: f32,
    pub toggle_enable_prob: f32,

    // Reproduction parameters
    pub crossover_rate: f32,
    pub survival_threshold: f32,

    // Speciation parameters
    pub species_elitism: bool,
    pub elitism: usize,
    pub stagnation_limit: usize,
    pub target_species_count: usize,

    // Allowed Activation Functions (incl. custom)
    pub allowed_activation_functions: Vec<ActivationFunction>,
    pub default_activation_function: ActivationFunction,
}

impl NeatConfig {
    pub fn default() -> Self {
        NeatConfig {
            network_type: NetworkType::Feedforward,
            bias_mutation_prob: 0.3,
            time_constant_mutation_prob: 0.2,
            param_perturb_prob: 0.9,

            population_size: 150,

            initial_compatibility_threshold: 3.0,
            compatibility_disjoint_coefficient: 1.0,
            compatibility_weight_coefficient: 0.3,

            weight_mutation_prob: 0.8,
            weight_perturb_prob: 0.9,
            new_connection_prob: 0.05,
            new_node_prob: 0.03,
            toggle_enable_prob: 0.01,

            crossover_rate: 0.75,
            survival_threshold: 0.2,

            species_elitism: true,
            elitism: 1,
            stagnation_limit: 35,
            target_species_count: 15,

            allowed_activation_functions: vec![ActivationFunction::Sigmoid],
            default_activation_function: ActivationFunction::Sigmoid,
        }
    }
}
