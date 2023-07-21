use crate::activation::Activation;
use crate::innovation_record::InnovationRecord;

pub struct Network {
    input_num: usize,
    output_num: usize,
    activation_function: Activation,
    innovation_record: InnovationRecord,
}

impl Network {
    pub fn new(input_num: usize, output_num: usize, activation_function: Activation) -> Self {
        Self {
            input_num,
            output_num,
            activation_function,
            innovation_record: InnovationRecord::new(input_num + output_num),
        }
    }
}
