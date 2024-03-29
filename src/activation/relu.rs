use crate::{train::GraphExecTrain, GraphExec};
use ndarray::{Array, Dimension, LinalgScalar, ScalarOperand};
use num_traits::Float;

use super::Activation;

#[derive(Debug, Copy, Clone)]
pub struct Relu;
impl Activation for Relu {}

impl<F, D> GraphExec<Array<F, D>> for Relu
where
    F: LinalgScalar + Float,
    D: Dimension,
{
    type Output = Array<F, D>;
    fn exec(&self, input: Array<F, D>) -> Self::Output {
        let zero = F::zero();
        input.mapv(|x| x.max(zero))
    }
}

impl<F, D> GraphExecTrain<Array<F, D>> for Relu
where
    F: LinalgScalar + ScalarOperand + Float,
    D: Dimension,
{
    type State = (Array<F, D>, Self::Output);
    fn forward(&self, input: Array<F, D>) -> (Self::State, Self::Output) {
        let output = self.exec(input.clone());
        ((input, output.clone()), output)
    }

    fn back(&self, (input, output): Self::State, d_output: Self::Output) -> (Array<F, D>, Self) {
        let d_input: Array<F, D> = output / input * d_output;
        (d_input, Self)
    }
}
