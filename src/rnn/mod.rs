use std::marker::PhantomData;

use crate::{
    activation::{Activation, Linear, softmax::Softmax, tanh::Tanh},
    array::{compact_front, dot_front, dot_inner},
    initialisers::Initialiser,
    train::GraphExecTrain,
    Graph, GraphExec, Mappable, Shaped, HDF5, dense::{Dense, DenseState},
};
use hdf5::H5Type;
use ndarray::{
    Array, Array1, Array2, ArrayBase, Axis, Data, Dim, DimMax, Dimension, Ix1, Ix2, LinalgScalar,
    RemoveAxis, ScalarOperand, s,
};
use num_traits::{Float, FromPrimitive, One, Zero};
use rand::{distributions::Distribution, Rng};

#[derive(Debug, Copy, Clone)]
pub struct SimpleRNN<I> {
    output_size: usize,
    neuron_size: usize,
    initialiser: I,
}

pub struct SimpleRNNSize<I> {
    output_size: usize,
    neuron_size: usize,
    initialiser: PhantomData<I>,
}

impl<I> SimpleRNN<I> {
    #[must_use]
    pub const fn output_size(output_size: usize) -> SimpleRNNSize<I> {
        SimpleRNNSize {
            output_size,
            neuron_size: output_size,
            initialiser: PhantomData,
        }
    }
}

impl<I> SimpleRNNSize<I> {
    pub const fn with_initialiser(self, initialiser: I) -> SimpleRNN<I> {
        SimpleRNN {
            output_size: self.output_size,
            neuron_size: self.neuron_size,
            initialiser,
        }
    }
}

impl<I, F> Graph<F, [usize; 2]> for SimpleRNN<I>
where
    I: Initialiser<F, (usize, usize)> + Clone,
{
    type State = SimpleRNNState<F>;
    type OutputShape = usize;

    fn get_output_shape(&self) -> usize {
        self.output_size
    }

    fn init_with_random(self, rng: &mut impl Rng, input_size: [usize; 2]) -> Self::State {
        let [_, input_size] = input_size;

        // let d = self
        //     .initialiser.clone()
        //     .into_distribution((input_size, self.output_size));

        // let u = Array2::from_shape_simple_fn((self.neuron_size, input_size), || d.sample(rng));
        // let w =
        //     Array2::from_shape_simple_fn((self.neuron_size, self.neuron_size), || d.sample(rng));
        // let b = Array1::from_shape_simple_fn(self.neuron_size, || d.sample(rng));

        let inner = Dense::output_size(self.neuron_size).with_initialiser(self.initialiser.clone()).init_with_random(rng, input_size);
        let out = Dense::output_size(self.output_size).with_initialiser(self.initialise).init_with_random(rng, self.neuron_size);

        SimpleRNNState { inner, out }
    }
}

#[derive(Debug, Clone)]
pub struct SimpleRNNState<F> {
    pub inner: Linear<DenseState<F>, Tanh>,
    pub out: Linear<DenseState<F>, Softmax>,
}

impl<F, S> GraphExec<ArrayBase<S, Ix2>> for SimpleRNNState<F>
where
    F: LinalgScalar + Float + ScalarOperand,
    S: Data<Elem = F>,
{
    type Output = Array<F, Ix2>;

    fn exec(&self, input: ArrayBase<S, Ix2>) -> Self::Output {
        let (timesteps, input_size) = input.raw_dim().into_pattern();
        let mut output = Array2::uninit([timesteps, self.c.raw_dim().size()]);

        let neuron_size = self.inner.graph.b.raw_dim().size();

        let mut h = Array1::<F>::zeros(neuron_size + input_size);
        let axis = Axis(0);
        for (xi, yi) in input.axis_iter(axis).zip(output.axis_iter_mut(axis)) {
            xi.assign_to(h.slice_mut(s![neuron_size..]));

            let h1 = self.inner.exec(&h);
            self.out.exec(h1).assign_to(yi);

            h1.assign_to(h.slice_mut(s![..neuron_size]));
        }

        unsafe { output.assume_init() }
    }
}

// impl<F, D> GraphExecTrain<Array<F, D>> for SimpleRNNState<F>
// where
//     F: LinalgScalar + FromPrimitive + ScalarOperand,
//     D: Dimension + DimMax<Ix1, Output = D> + RemoveAxis,
// {
//     type State = Self::Output;
//     fn forward(&self, input: Array<F, D>) -> (Self::State, Self::Output) {
//         (input.clone(), self.exec(input))
//     }

//     fn back(&self, input: Self::State, d_output: Self::Output) -> (Array<F, D>, Self) {
//         let di = dot_inner(d_output.clone(), &self.w.t());
//         let db = compact_front(d_output.clone()).mean_axis(Axis(0)).unwrap();
//         let dw = dot_front(input, d_output);
//         (di, Self { w: dw, b: db })
//     }
// }

// impl<T> Mappable<T> for SimpleRNNState<T> {
//     // not redundant. just forces a capture without needing to clone
//     #![allow(clippy::redundant_closure)]

//     fn map<F: FnMut(&T) -> T>(&self, mut f: F) -> Self {
//         let SimpleRNNState { w, b } = self;
//         let w = w.map(|a| f(a));
//         let b = b.map(f);
//         Self { w, b }
//     }
//     fn map_mut<F: FnMut(&mut T)>(&mut self, mut f: F) {
//         self.w.map_mut(|a| f(a));
//         self.b.map_mut(f);
//     }
//     fn map_mut_with<F: FnMut(&mut T, &T)>(&mut self, rhs: &Self, mut f: F) {
//         self.w.zip_mut_with(&rhs.w, |a, b| f(a, b));
//         self.b.zip_mut_with(&rhs.b, f);
//     }
// }

// impl<T> Shaped<T> for SimpleRNNState<T>
// where
//     T: Clone + Zero + One,
// {
//     type Shape = Dim<[usize; 2]>;
//     fn shape(&self) -> Self::Shape {
//         self.w.raw_dim()
//     }
//     fn zero(shape: Self::Shape) -> Self {
//         Self {
//             w: Array2::zeros(shape),
//             b: Array1::zeros(shape[1]),
//         }
//     }
//     fn one(shape: Self::Shape) -> Self {
//         Self {
//             w: Array2::ones(shape),
//             b: Array1::ones(shape[1]),
//         }
//     }
//     fn iter(shape: Self::Shape, mut i: impl Iterator<Item = T>) -> Self {
//         Self {
//             w: Array2::from_shape_fn(shape, |_| i.next().unwrap()),
//             b: Array1::from_shape_fn(shape[1], |_| i.next().unwrap()),
//         }
//     }
// }

// impl<F: H5Type, I> HDF5<F, usize> for SimpleRNN<I>
// where
//     I: Initialiser<F, (usize, usize)>,
// {
//     fn save(&self, state: &Self::State, group: &hdf5::Group) -> hdf5::Result<()> {
//         group
//             .new_dataset_builder()
//             .with_data(state.w.view())
//             .create("weights")?;
//         group
//             .new_dataset_builder()
//             .with_data(state.b.view())
//             .create("bias")?;
//         Ok(())
//     }

//     fn load(&self, group: &hdf5::Group) -> hdf5::Result<Self::State> {
//         let w = group.dataset("weights")?.read()?;
//         let b = group.dataset("bias")?.read()?;

//         Ok(SimpleRNNState { w, b })
//     }
// }
