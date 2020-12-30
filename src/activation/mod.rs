use crate::{Graph, GraphExec, GraphExecTrain, Mappable};
use rand::Rng;

pub mod relu;
pub mod sigmoid;

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

impl<T, G, L> Mappable<T> for Linear<G, L>
where
    G: Mappable<T>,
    L: Clone,
{
    fn map<F: FnMut(&T) -> T + Clone>(&self, f: F) -> Self {
        Linear {
            graph: self.graph.map(f),
            linear: self.linear.clone(),
        }
    }
    fn map_mut<F: FnMut(&mut T) + Clone>(&mut self, f: F) {
        self.graph.map_mut(f)
    }
    fn map_mut_with<F: FnMut(&mut T, &T) + Clone>(&mut self, rhs: &Self, f: F) {
        self.graph.map_mut_with(&rhs.graph, f)
    }
}
