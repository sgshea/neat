use serde::{Deserialize, Serialize};

use crate::nn::nn::NetworkType;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    Identity,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
}

impl ActivationFunction {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Identity => x,
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Relu => x.max(0.0),
            ActivationFunction::LeakyRelu => x.max(0.01 * x),
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
    // Default Activation Function used for hidden nodes
    pub default_activation_function: ActivationFunction,
    // Activation Function used for input nodes
    pub input_activation_function: ActivationFunction,
    // Activation Function used for output nodes
    pub output_activation_function: ActivationFunction,

    // Pressure to minimize structure (Parsimony)
    pub complexity_penalty_coefficient: f32,
    pub connections_penalty_coefficient: f32,
    pub target_complexity: usize,
    pub complexity_threshold: usize,
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
            input_activation_function: ActivationFunction::Identity,
            output_activation_function: ActivationFunction::Identity,
            complexity_penalty_coefficient: 0.001,
            connections_penalty_coefficient: 0.0005,
            target_complexity: 7,
            complexity_threshold: 10,
        }
    }

    // Create a builder with default values
    pub fn builder() -> NeatConfigBuilder {
        NeatConfigBuilder::default()
    }
}

// Builder pattern implementation
#[derive(Debug, Clone)]
pub struct NeatConfigBuilder {
    config: NeatConfig,
}

impl Default for NeatConfigBuilder {
    fn default() -> Self {
        Self {
            config: NeatConfig::default(),
        }
    }
}

// Group methods by category for better organization
impl NeatConfigBuilder {
    // Network type
    pub fn network_type(mut self, network_type: NetworkType) -> Self {
        self.config.network_type = network_type;
        self
    }

    // CTRNN specific parameters
    pub fn bias_mutation_prob(mut self, prob: f32) -> Self {
        self.config.bias_mutation_prob = prob;
        self
    }

    pub fn time_constant_mutation_prob(mut self, prob: f32) -> Self {
        self.config.time_constant_mutation_prob = prob;
        self
    }

    pub fn param_perturb_prob(mut self, prob: f32) -> Self {
        self.config.param_perturb_prob = prob;
        self
    }

    // General parameters
    pub fn population_size(mut self, size: usize) -> Self {
        self.config.population_size = size;
        self
    }

    // Compatibility parameters
    pub fn compatibility(mut self, threshold: f32, disjoint_coef: f32, weight_coef: f32) -> Self {
        self.config.initial_compatibility_threshold = threshold;
        self.config.compatibility_disjoint_coefficient = disjoint_coef;
        self.config.compatibility_weight_coefficient = weight_coef;
        self
    }

    // Mutation parameters
    pub fn mutation_rates(
        mut self,
        weight_mutation: f32,
        weight_perturb: f32,
        new_connection: f32,
        new_node: f32,
        toggle_enable: f32,
    ) -> Self {
        self.config.weight_mutation_prob = weight_mutation;
        self.config.weight_perturb_prob = weight_perturb;
        self.config.new_connection_prob = new_connection;
        self.config.new_node_prob = new_node;
        self.config.toggle_enable_prob = toggle_enable;
        self
    }

    // Reproduction parameters
    pub fn reproduction(mut self, crossover_rate: f32, survival_threshold: f32) -> Self {
        self.config.crossover_rate = crossover_rate;
        self.config.survival_threshold = survival_threshold;
        self
    }

    // Speciation parameters
    pub fn speciation(
        mut self,
        species_elitism: bool,
        elitism: usize,
        stagnation_limit: usize,
        target_species_count: usize,
    ) -> Self {
        self.config.species_elitism = species_elitism;
        self.config.elitism = elitism;
        self.config.stagnation_limit = stagnation_limit;
        self.config.target_species_count = target_species_count;
        self
    }

    // Activation functions
    pub fn activation_functions(mut self, allowed: Vec<ActivationFunction>) -> Self {
        self.config.allowed_activation_functions = allowed;
        self
    }

    pub fn default_activation_function(mut self, default: ActivationFunction) -> Self {
        self.config.default_activation_function = default;
        self
    }

    pub fn input_activation_function(mut self, input: ActivationFunction) -> Self {
        self.config.input_activation_function = input;
        self
    }

    pub fn output_activation_function(mut self, output: ActivationFunction) -> Self {
        self.config.output_activation_function = output;
        self
    }

    // Complexity/parsimony parameters
    pub fn complexity_control(
        mut self,
        penalty_coef: f32,
        connections_penalty: f32,
        target: usize,
        threshold: usize,
    ) -> Self {
        self.config.complexity_penalty_coefficient = penalty_coef;
        self.config.connections_penalty_coefficient = connections_penalty;
        self.config.target_complexity = target;
        self.config.complexity_threshold = threshold;
        self
    }

    // Individual setters for fine-grained control
    pub fn with_weight_mutation_prob(mut self, prob: f32) -> Self {
        self.config.weight_mutation_prob = prob;
        self
    }

    pub fn with_new_connection_prob(mut self, prob: f32) -> Self {
        self.config.new_connection_prob = prob;
        self
    }

    pub fn with_new_node_prob(mut self, prob: f32) -> Self {
        self.config.new_node_prob = prob;
        self
    }

    // Build the final config
    pub fn build(self) -> NeatConfig {
        self.config
    }
}
