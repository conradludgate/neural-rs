pub mod activation;
pub mod cost;
pub mod dense;
pub mod initialisers;
pub mod optimise;
pub mod tuple;

use rand::Rng;
use std::ops::{Deref, DerefMut};

pub trait GraphExec<Input> {
    type Output;

    /// Executes the computation graph on the given input to create
    /// the output value
    fn exec(&self, input: &Input) -> Self::Output;
}

pub trait GraphExecTrain<Input>: GraphExec<Input> {
    type State;
    fn forward(&self, input: &Input) -> (Self::State, Self::Output);
    fn back(&self, state: Self::State, d_output: Self::Output) -> (Input, Self);
}

/// An abstract representation of a Computation Graph.
/// F is the inner float representation
pub trait Graph<F>: Sized {
    /// The state that this builder produces
    type State;

    /// Gets the graph's expected input size
    fn get_input_size(&self) -> usize;

    /// Initializes the graph
    fn init(self, output_size: usize) -> Self::State {
        let mut rng = rand::prelude::thread_rng();
        self.init_with_random(&mut rng, output_size)
    }

    /// Use to initialise with a predefined random source
    fn init_with_random(self, rng: &mut impl Rng, output_size: usize) -> Self::State;
}

pub trait Cost<T> {
    type Inner;
    fn cost(input: &T, expected: &T) -> Self::Inner;
    fn diff(input: &T, expected: &T) -> T;
}

pub trait Optimiser<G> {
    fn optimise(&mut self, graph: &mut G, grads: G);
}

pub struct Train<C, O, G> {
    optimiser: O,
    graph: G,
    _cost: C,
}

impl<C, O, G> Train<C, O, G> {
    pub fn new(graph: G, cost: C, optimiser: O) -> Self {
        Train {
            graph,
            _cost: cost,
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
        let (state, output): _ = self.graph.forward(input);

        let d_output: _ = C::diff(&output, expected);
        let (_, grads) = self.graph.back(state, d_output);

        self.optimiser.optimise(&mut self.graph, grads);

        C::cost(&output, expected)
    }
}

#[cfg(test)]
mod test {
    use activation::sigmoid::Sigmoid;
    use cost::MSE;
    use initialisers::Xavier;
    use ndarray::Array2;
    use optimise::SGD;

    use crate::dense::Dense;
    use crate::*;

    #[test]
    fn test_graph() {
        // Create a new compute graph which uses three Dense components
        // With the input having size 28*28 and the output having size 10
        // Initialise it with uniform random data
        let network: _ = tuple![
            Dense::new(28 * 28, Xavier).with_activation(Sigmoid),
            Dense::new(16, Xavier).with_activation(Sigmoid),
            Dense::new(16, Xavier).with_activation(Sigmoid)
        ]
        .init(10);

        // New trainer with mean squared error cost function and
        // stochastic gradient descent optimisation (alpha=0.01)
        let mut trainer: _ = Train::new(network, MSE, SGD::new(0.01));

        let mut costs = vec![];
        for _ in 1..100 {
            let input: _ = Array2::<f64>::zeros((28 * 28, 100));
            let expected: _ = Array2::<f64>::zeros((10, 100));

            // Train the network on the given input the respective expected value
            let cost = trainer.train(&input, &expected);
            costs.push(cost);
        }
        let network: _ = trainer.into_inner();

        // Finally, use the network
        let input = Array2::<f64>::zeros((28 * 28, 100));
        let output = network.exec(&input);
        assert_eq!(output.shape(), &[10, 100]);
    }
}
