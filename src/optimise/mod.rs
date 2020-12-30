pub mod sgd;
use std::ops::{Deref, DerefMut};

use crate::{Cost, GraphExecTrain};

pub trait Optimiser<G> {
    fn optimise(&mut self, graph: &mut G, grads: G);
}

pub struct Train<C, O, G> {
    optimiser: O,
    graph: G,
    cost: C,
}

impl<C, O, G> Train<C, O, G> {
    pub fn new(graph: G, cost: C, optimiser: O) -> Self {
        Train {
            graph,
            cost: cost,
            optimiser,
        }
    }
    pub fn into_inner(self) -> G {
        self.graph
    }
}

impl<C, O, G> Deref for Train<C, O, G> {
    type Target = G;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}
impl<C, O, G> DerefMut for Train<C, O, G> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<C, O, G> Train<C, O, G> {
    pub fn train<Input>(&mut self, input: &Input, expected: &G::Output) -> C::Inner
    where
        C: Cost<G::Output>,
        O: Optimiser<G>,
        G: GraphExecTrain<Input>,
    {
        let (grads, cost) = self.graph.get_grads(input, expected, &self.cost);
        self.optimiser.optimise(&mut self.graph, grads);
        cost
    }
}
