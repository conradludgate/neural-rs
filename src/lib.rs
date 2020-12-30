pub mod activation;
pub mod cost;
pub mod dense;
pub mod derivative;
pub mod initialisers;
pub mod optimise;
pub mod tuple;

use rand::Rng;

pub trait GraphExec<Input> {
    type Output;

    /// Executes the computation graph on the given input to create
    /// the output value
    fn exec(&self, input: &Input) -> Self::Output;
}

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

/// An abstract representation of a Computation Graph.
/// F is the inner float representation
pub trait Graph<F, InputShape>: Sized {
    /// The state that this builder produces
    type State;
    type OutputShape;

    /// Gets the graph's expected input size
    fn get_output_shape(&self) -> Self::OutputShape;

    /// Initializes the graph
    fn input_shape(self, input_shape: InputShape) -> Self::State {
        let mut rng = rand::prelude::thread_rng();
        self.init_with_random(&mut rng, input_shape)
    }

    /// Use to initialise with a predefined random source
    fn init_with_random(self, rng: &mut impl Rng, input_shape: InputShape) -> Self::State;
}

pub trait Cost<T> {
    type Inner;
    fn cost(&self, output: &T, expected: &T) -> Self::Inner;
    fn diff(&self, output: &T, expected: &T) -> T;
}

// #[cfg(test)]
// mod test {
//     use activation::{relu::Relu, sigmoid::Sigmoid};
//     use cost::MSE;
//     use derivative::get_grads;
//     use initialisers::Xavier;
//     use ndarray::Array2;
//     use optimise::{sgd::SGD, Optimiser};
//     use rand::thread_rng;

//     use crate::dense::Dense;
//     use crate::*;

//     #[test]
//     fn test_grads() {
//         let mut network: _ = tuple![
//             Dense::new(6, Xavier).with_activation(Sigmoid),
//             Dense::new(4, Xavier).with_activation(Relu)
//         ]
//         .input_shape(8);

//         let mut rng = thread_rng();

//         let input: _ = Array2::<f64>::from_shape_fn((8, 10), |_| rng.gen());
//         let expected: _ = Array2::<f64>::from_shape_fn((4, 10), |_| rng.gen());

//         let grads: _ = network.get_grads(&input, &expected, &MSE).0;
//         let expected_grads: _ = get_grads(&mut network, MSE, 1e-8, &input, &expected);

//         println!("{:?}", expected_grads);
//         println!("{:?}", grads);
//         panic!();
//     }

//     #[test]
//     fn test_train() {
//         let mut network: _ = tuple![
//             Dense::new(6, Xavier).with_activation(Sigmoid),
//             Dense::new(4, Xavier).with_activation(Sigmoid)
//         ]
//         .input_shape(8);

//         let mut rng = thread_rng();

//         let input: _ = Array2::<f64>::from_shape_fn((8, 10), |_| rng.gen());
//         let expected: _ = Array2::<f64>::from_shape_fn((4, 10), |_| rng.gen());

//         let cost1 = MSE.cost(&network.exec(&input), &expected);
//         let grads: _ = get_grads(&mut network, MSE, 1e-8, &input, &expected);
//         let mut optimiser = SGD::new(0.1);
//         optimiser.optimise(&mut network, grads);
//         let cost2 = MSE.cost(&network.exec(&input), &expected);

//         println!("cost1 {:?}, cost2: {:?}", cost1, cost2);
//         panic!();
//     }
// }
