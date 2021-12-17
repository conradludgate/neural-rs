use crate::{train::GraphExecTrain, GraphExec};
use ndarray::{Array, Dimension, LinalgScalar, ScalarOperand};
use num_traits::Float;

use super::Activation;

#[derive(Debug, Copy, Clone)]
pub struct Sigmoid;
impl Activation for Sigmoid {}

impl<F, D> GraphExec<Array<F, D>> for Sigmoid
where
    F: LinalgScalar + Float,
    D: Dimension,
{
    type Output = Array<F, D>;
    fn exec(&self, input: Array<F, D>) -> Self::Output {
        let one = F::one();
        input.mapv(|x| (one / (one + (-x).exp())))
    }
}

impl<F, D> GraphExecTrain<Array<F, D>> for Sigmoid
where
    F: LinalgScalar + ScalarOperand + Float,
    D: Dimension,
{
    type State = Self::Output;
    fn forward(&self, input: Array<F, D>) -> (Self::State, Self::Output) {
        let output = self.exec(input);
        (output.clone(), output)
    }

    fn back(&self, output: Self::State, d_output: Self::Output) -> (Array<F, D>, Self) {
        let d_input: Array<F, D> = d_output * &output * (-output + F::one());
        (d_input, Self)
    }
}
