use crate::{train::GraphExecTrain, GraphExec};
use ndarray::{Array, Dimension, LinalgScalar};
use num_traits::{Float, Inv};

use super::Activation;

#[derive(Debug, Copy, Clone)]
pub struct Tanh;
impl Activation for Tanh {}

impl<F, D> GraphExec<Array<F, D>> for Tanh
where
    F: LinalgScalar + Float,
    D: Dimension,
{
    type Output = Array<F, D>;
    fn exec(&self, input: Array<F, D>) -> Self::Output {
        input.mapv_into(F::tanh)
    }
}

impl<F, D> GraphExecTrain<Array<F, D>> for Tanh
where
    F: LinalgScalar + Inv<Output = F> + Float,
    D: Dimension,
{
    type State = Array<F, D>;
    fn forward(&self, input: Array<F, D>) -> (Self::State, Self::Output) {
        let output = self.exec(input.clone());
        (input, output)
    }

    fn back(&self, input: Self::State, d_output: Self::Output) -> (Array<F, D>, Self) {
        // derivative of tanh is sech^2 which is 1/cosh^2
        (d_output * input.mapv_into(|x| x.cosh().powi(2).inv()), Self)
    }
}
