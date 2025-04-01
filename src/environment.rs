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
