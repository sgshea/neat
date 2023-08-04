use crate::individual::Individual;

// User implements environment for their problem to test fitness
pub trait Environment {
    fn evaluate(&mut self, individual: &mut Individual);
}
