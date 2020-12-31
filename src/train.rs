use std::ops::{Deref, DerefMut};

use num_traits::Float;
use rand::prelude::*;
use rand_distr::{
    uniform::{SampleBorrow, SampleUniform},
    Uniform,
};

use crate::{cost::Cost, optimise::Optimiser, GraphExec, Mappable, Shaped};

pub trait GraphExecTrain<Input>: GraphExec<Input> + Sized {
    type State;
    fn forward(&self, input: &Input) -> (Self::State, Self::Output);
    fn back(&self, state: Self::State, d_output: Self::Output) -> (Input, Self);
    fn get_grads<C>(&self, input: &Input, expected: &Self::Output, cost: &C) -> (Self, C::Inner)
    where
        C: Cost<Self::Output>,
    {
        let (state, output): _ = self.forward(input);

        let d_output: _ = cost.diff(&output, &expected);
        (self.back(state, d_output).1, cost.cost(&output, &expected))
    }
}

pub struct Train<F, C, O, G> {
    pub graph: G,
    pub optimiser: O,
    pub cost: C,
    pub regularisation: Option<Regularisation<F>>,
    pub dropout: F,
}

impl<F, C, O, G> Deref for Train<F, C, O, G> {
    type Target = G;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}
impl<F, C, O, G> DerefMut for Train<F, C, O, G> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<F, C, O, G> Train<F, C, O, G> {
    pub fn train<Input>(&mut self, input: &Input, expected: &G::Output) -> C::Inner
    where
        C: Cost<G::Output, Inner = F>,
        O: Optimiser<G>,
        G: GraphExecTrain<Input> + Mappable<F> + Shaped<F> + Clone,
        F: Float + SampleBorrow<F> + SampleUniform,
    {
        let zero = F::zero();
        let one = F::one();

        let (mut grads, mut cost) = if (zero..one).contains(&self.dropout) {
            let dropouts = G::iter(
                self.graph.shape(),
                thread_rng().sample_iter(Uniform::new_inclusive(zero, one)),
            );

            let a = one / (one - self.dropout);
            let mut graph = self.graph.clone();
            graph.map_mut_with(&dropouts, |g, &d| {
                if d < self.dropout {
                    *g = zero;
                } else {
                    *g = *g * a;
                }
            });

            let (mut grads, cost) = self.graph.get_grads(input, expected, &self.cost);

            grads.map_mut_with(&dropouts, |g, &d| {
                if d < self.dropout {
                    *g = zero;
                } else {
                    *g = *g * a;
                }
            });

            (grads, cost)
        } else {
            self.graph.get_grads(input, expected, &self.cost)
        };

        if let Some(r) = self.regularisation {
            cost = cost + r.apply(&mut grads, &self.graph);
        }

        self.optimiser.optimise(&mut self.graph, grads);
        cost
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Regularisation<F> {
    L1(F),
    L2(F),
    L1_2(F, F),
}

impl<F> Regularisation<F>
where
    F: Float,
{
    fn apply<G: Mappable<F>>(self, grads: &mut G, graph: &G) -> F {
        let mut cost = F::zero();
        match self {
            Regularisation::L1(a) => {
                grads.map_mut_with(&graph, move |g, &x| {
                    cost = cost + x.abs() * a;
                    *g = *g + x.signum() * a;
                });
            }
            Regularisation::L2(a) => {
                grads.map_mut_with(&graph, move |g, &x| {
                    cost = cost + x * x * a;
                    *g = *g + (x + x) * a;
                });
            }
            Regularisation::L1_2(a, b) => {
                grads.map_mut_with(&graph, move |g, &x| {
                    cost = cost + x.abs() * a + x * x * b;
                    *g = *g + x.signum() * a + (x + x) * b;
                });
            }
        }
        cost
    }
}
