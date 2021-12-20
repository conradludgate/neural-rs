use crate::{train::GraphExecTrain, GraphExec};
use ndarray::{Array, Dimension, LinalgScalar, ScalarOperand};
use num_traits::Float;

use super::Activation;

#[derive(Debug, Copy, Clone)]
pub struct Softmax;
impl Activation for Softmax {}

impl<F, D> GraphExec<Array<F, D>> for Softmax
where
    F: LinalgScalar + Float,
    D: Dimension,
{
    type Output = Array<F, D>;
    fn exec(&self, input: Array<F, D>) -> Self::Output {
        let y = input.mapv_into(F::exp);
        y / y.sum()
    }
}

impl<F, D> GraphExecTrain<Array<F, D>> for Softmax
where
    F: LinalgScalar + ScalarOperand + Float,
    D: Dimension,
{
    type State = (Self::Output, F);
    fn forward(&self, input: Array<F, D>) -> (Self::State, Self::Output) {
        let y = input.mapv_into(F::exp);
        let s = y.sum();
        let output = y / s;
        ((output.clone(), s), output)
    }

    fn back(&self, (output, s): Self::State, d_output: Self::Output) -> (Array<F, D>, Self) {
        let d_input: Array<F, D> = d_output * (output - (F::one() / s.powi(2)));
        (d_input, Self)
    }
}
