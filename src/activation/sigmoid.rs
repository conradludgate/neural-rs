use std::ops::{Add, AddAssign, Mul};

use crate::{Graph, GraphExec, GraphExecTrain};
use ndarray::{Array, Dimension, LinalgScalar, ScalarOperand};
use num_traits::Float;
use rand::Rng;

use super::Activation;

#[derive(Debug, Copy, Clone)]
pub struct Sigmoid;
impl<G> Activation<G> for Sigmoid {
    type Activation = SigmoidGraph<G>;
    fn into_activation(self, g: G) -> Self::Activation {
        SigmoidGraph::new(g)
    }
}
#[derive(Debug, Copy, Clone)]
pub struct SigmoidGraph<G>(G);
impl<G> SigmoidGraph<G> {
    pub fn new(g: G) -> Self {
        SigmoidGraph(g)
    }
}

impl<F, G> Graph<F> for SigmoidGraph<G>
where
    G: Graph<F>,
{
    type State = SigmoidGraph<G::State>;

    fn get_input_size(&self) -> usize {
        self.0.get_input_size()
    }

    fn init_with_random(self, rng: &mut impl Rng, output_size: usize) -> Self::State {
        SigmoidGraph(self.0.init_with_random(rng, output_size))
    }
}

impl<F, D, G, Input> GraphExec<Input> for SigmoidGraph<G>
where
    G: GraphExec<Input, Output = Array<F, D>>,
    F: LinalgScalar + Float,
    D: Dimension,
{
    type Output = Array<F, D>;
    fn exec(&self, input: &Input) -> Self::Output {
        let mut input = self.0.exec(&input);
        let one = F::one();
        input.mapv_inplace(|x| (one / (one + x.exp())));
        input
    }
}

impl<F, D, G, Input> GraphExecTrain<Input> for SigmoidGraph<G>
where
    G: GraphExecTrain<Input, Output = Array<F, D>>,
    F: LinalgScalar + ScalarOperand + Float,
    D: Dimension,
{
    type State = (G::State, Self::Output);
    fn forward(&self, input: &Input) -> (Self::State, Self::Output) {
        let (state, mut input) = self.0.forward(&input);

        let one = F::one();
        input.mapv_inplace(|x| (one / (one + x.exp())));
        let output = input;

        ((state, output.clone()), output)
    }

    fn back(&self, state: Self::State, d_output: Self::Output) -> (Input, Self) {
        let (state, output) = state;
        let d_output: Array<F, D> = d_output * &output * (-output + F::one());
        let (d_input, grads) = self.0.back(state, d_output);
        (d_input, SigmoidGraph(grads))
    }
}

impl<T, U> Add<SigmoidGraph<U>> for SigmoidGraph<T>
where
    T: Add<U>,
{
    type Output = SigmoidGraph<T::Output>;
    fn add(self, rhs: SigmoidGraph<U>) -> Self::Output {
        SigmoidGraph(self.0 + rhs.0)
    }
}
impl<T, U> AddAssign<SigmoidGraph<U>> for SigmoidGraph<T>
where
    T: AddAssign<U>,
{
    fn add_assign(&mut self, rhs: SigmoidGraph<U>) {
        self.0 += rhs.0;
    }
}

impl<T, U> Mul<SigmoidGraph<U>> for SigmoidGraph<T>
where
    T: Mul<U>,
{
    type Output = SigmoidGraph<T::Output>;
    fn mul(self, rhs: SigmoidGraph<U>) -> Self::Output {
        SigmoidGraph(self.0 * rhs.0)
    }
}

impl<T, S> Mul<S> for SigmoidGraph<T>
where
    S: ScalarOperand,
    T: Mul<S>,
{
    type Output = SigmoidGraph<T::Output>;
    fn mul(self, rhs: S) -> Self::Output {
        SigmoidGraph(self.0 * rhs)
    }
}
