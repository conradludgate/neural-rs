use super::Cost;
use ndarray::{Array1, Array2, LinalgScalar, ScalarOperand};
use num_traits::FromPrimitive;

#[derive(Debug, Copy, Clone)]
pub struct MSE;

impl<F> Cost<Array1<F>> for MSE
where
    F: LinalgScalar + ScalarOperand,
{
    type Inner = F;
    fn cost(&self, input: &Array1<F>, expected: &Array1<F>) -> Self::Inner {
        let diff = input - expected;
        diff.dot(&diff)
    }
    fn diff(&self, input: &Array1<F>, expected: &Array1<F>) -> Array1<F> {
        let one = F::one();
        let two = one + one;
        (input - expected) * two
    }
}

impl<F> Cost<Array2<F>> for MSE
where
    F: LinalgScalar + ScalarOperand + FromPrimitive,
{
    type Inner = F;
    fn cost(&self, input: &Array2<F>, expected: &Array2<F>) -> Self::Inner {
        let diff = input - expected;
        diff.t().dot(&diff).mean().unwrap()
    }
    fn diff(&self, input: &Array2<F>, expected: &Array2<F>) -> Array2<F> {
        let one = F::one();
        let two = one + one;
        (input - expected) * two
    }
}
