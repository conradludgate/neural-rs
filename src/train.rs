use std::ops::{Deref, DerefMut};

use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis};
use num_traits::{Float, FromPrimitive};
use rand::prelude::*;
use rand_distr::{
    uniform::{SampleBorrow, SampleUniform},
    Uniform,
};

use crate::{cost::Cost, optimise::Optimiser, GraphExec, Mappable, Shaped};

pub trait GraphExecTrain<Input>: GraphExec<Input> + Sized {
    type State;
    fn forward(&self, input: Input) -> (Self::State, Self::Output);
    fn back(&self, state: Self::State, d_output: Self::Output) -> (Input, Self);
    fn get_grads<C>(&self, input: Input, expected: Self::Output, cost: &C) -> (Self, C::Inner)
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
    pub fn perform_epoch<D1, D2>(
        &mut self,
        inputs: ArrayView<F, D1>,
        expected: ArrayView<F, D2>,
        batch_size: usize,
    ) -> C::Inner
    where
        C: Cost<G::Output, Inner = F>,
        O: Optimiser<G>,
        G: GraphExecTrain<Array<F, D1>, Output = Array<F, D2>> + Mappable<F> + Shaped<F> + Clone,
        F: Float + SampleBorrow<F> + SampleUniform + Clone + FromPrimitive,
        D1: Dimension + RemoveAxis,
        D2: Dimension + RemoveAxis,
    {
        assert_eq!(inputs.raw_dim()[0], expected.raw_dim()[0]);
        let total_inputs = inputs.raw_dim()[0];

        let mut rng = thread_rng();
        let mut indicies: Vec<_> = (0..total_inputs).collect();
        indicies.shuffle(&mut rng);

        let mut cost = F::zero();
        for i in (0..total_inputs).step_by(batch_size) {
            cost = cost + self.train_batch(&inputs, &expected, &indicies[i..i + batch_size]);
        }
        if total_inputs % batch_size != 0 {
            let i = total_inputs - total_inputs % batch_size;
            cost = cost + self.train_batch(&inputs, &expected, &indicies[i..total_inputs]);
        }

        cost / F::from_usize((total_inputs + batch_size - 1) / batch_size).unwrap()
    }

    pub fn train_batch<D1, D2>(
        &mut self,
        inputs: &ArrayView<F, D1>,
        expected: &ArrayView<F, D2>,
        indicies: &[usize],
    ) -> C::Inner
    where
        C: Cost<G::Output, Inner = F>,
        O: Optimiser<G>,
        G: GraphExecTrain<Array<F, D1>, Output = Array<F, D2>> + Mappable<F> + Shaped<F> + Clone,
        F: Float + SampleBorrow<F> + SampleUniform + Clone,
        D1: Dimension + RemoveAxis,
        D2: Dimension + RemoveAxis,
    {
        unsafe {
            let mut input_dim = inputs.raw_dim();
            input_dim.as_array_view_mut()[0] = indicies.len();

            let mut expected_dim = expected.raw_dim();
            expected_dim.as_array_view_mut()[0] = indicies.len();

            let mut shuffled_inputs = Array::uninitialized(input_dim);
            let mut shuffled_expected = Array::uninitialized(expected_dim);

            for (i, &j) in indicies.into_iter().enumerate() {
                shuffled_inputs
                    .index_axis_mut(Axis(0), i)
                    .assign(&inputs.index_axis(Axis(0), j));

                shuffled_expected
                    .index_axis_mut(Axis(0), i)
                    .assign(&expected.index_axis(Axis(0), j));
            }

            self.train(shuffled_inputs, shuffled_expected)
        }
    }

    pub fn train<D1, D2>(&mut self, input: Array<F, D1>, expected: Array<F, D2>) -> C::Inner
    where
        C: Cost<G::Output, Inner = F>,
        O: Optimiser<G>,
        G: GraphExecTrain<Array<F, D1>, Output = Array<F, D2>> + Mappable<F> + Shaped<F> + Clone,
        F: Float + SampleBorrow<F> + SampleUniform + Clone,
        D1: Dimension,
        D2: Dimension,
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
