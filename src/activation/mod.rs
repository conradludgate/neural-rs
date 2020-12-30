use std::ops::{Add, AddAssign, Mul};

use crate::{derivative::DerivativeTesting, Graph, GraphExec, GraphExecTrain};
use ndarray::ScalarOperand;
use rand::Rng;

pub mod sigmoid;
pub mod relu;

pub trait Activation<G>: Sized {
    fn into_activation(self, g: G) -> Linear<G, Self>;
}

#[derive(Debug, Copy, Clone)]
pub struct Linear<G, L> {
    graph: G,
    linear: L,
}

impl<G, L> Linear<G, L> {
    pub fn new(graph: G, linear: L) -> Self {
        Linear { graph, linear }
    }
}

impl<I, G, F, L> Graph<F, I> for Linear<G, L>
where
    G: Graph<F, I>,
{
    type State = Linear<G::State, L>;
    type OutputShape = G::OutputShape;

    fn get_output_shape(&self) -> Self::OutputShape {
        self.graph.get_output_shape()
    }

    fn init_with_random(self, rng: &mut impl Rng, input_shape: I) -> Self::State {
        let Linear { graph, linear } = self;
        Linear {
            graph: graph.init_with_random(rng, input_shape),
            linear,
        }
    }
}

impl<G, L, Input> GraphExec<Input> for Linear<G, L>
where
    G: GraphExec<Input>,
    L: GraphExec<G::Output, Output = G::Output>,
{
    type Output = G::Output;
    fn exec(&self, input: &Input) -> Self::Output {
        let output = self.graph.exec(&input);
        self.linear.exec(&output)
    }
}

impl<G, L, Input> GraphExecTrain<Input> for Linear<G, L>
where
    G: GraphExecTrain<Input>,
    L: GraphExecTrain<G::Output, Output = G::Output>,
{
    type State = Linear<G::State, L::State>;
    fn forward(&self, input: &Input) -> (Self::State, Self::Output) {
        let (graph, output): _ = self.graph.forward(&input);
        let (linear, output): _ = self.linear.forward(&output);
        (Linear { graph, linear }, output)
    }

    fn back(&self, state: Self::State, d_output: Self::Output) -> (Input, Self) {
        let Linear { linear, graph } = state;
        let (d_output, linear): _ = self.linear.back(linear, d_output);
        let (d_input, graph): _ = self.graph.back(graph, d_output);
        (d_input, Linear { linear, graph })
    }
}

impl<T, U, L> Add<Linear<U, L>> for Linear<T, L>
where
    T: Add<U>,
{
    type Output = Linear<T::Output, L>;
    fn add(self, rhs: Linear<U, L>) -> Self::Output {
        Linear {
            graph: self.graph + rhs.graph,
            linear: self.linear,
        }
    }
}
impl<T, U, L> AddAssign<Linear<U, L>> for Linear<T, L>
where
    T: AddAssign<U>,
{
    fn add_assign(&mut self, rhs: Linear<U, L>) {
        self.graph += rhs.graph;
    }
}
impl<T, U, L> Mul<Linear<U, L>> for Linear<T, L>
where
    T: Mul<U>,
{
    type Output = Linear<T::Output, L>;
    fn mul(self, rhs: Linear<U, L>) -> Self::Output {
        Linear {
            graph: self.graph * rhs.graph,
            linear: self.linear,
        }
    }
}
impl<T, L, S> Mul<S> for Linear<T, L>
where
    T: Mul<S>,
    S: ScalarOperand,
{
    type Output = Linear<T::Output, L>;
    fn mul(self, rhs: S) -> Self::Output {
        Linear {
            graph: self.graph * rhs,
            linear: self.linear,
        }
    }
}
impl<T, U, L> PartialEq<Linear<U, L>> for Linear<T, L>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &Linear<U, L>) -> bool {
        self.graph == rhs.graph
    }
}

impl<F, G, L> DerivativeTesting<F> for Linear<G, L>
where
    G: DerivativeTesting<F>,
{
    fn len(&self) -> usize {
        self.graph.len()
    }
    fn get(&self, i: usize) -> F {
        self.graph.get(i)
    }

    fn set(&mut self, i: usize, f: F) {
        self.graph.set(i, f)
    }
}
