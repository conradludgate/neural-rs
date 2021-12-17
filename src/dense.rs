use std::marker::PhantomData;

use crate::{
    activation::{Activation, Linear},
    array::{compact_front, dot_front, dot_inner},
    initialisers::Initialiser,
    train::GraphExecTrain,
    Graph, GraphExec, Mappable, Shaped, HDF5,
};
use hdf5::H5Type;
use ndarray::{
    Array, Array1, Array2, ArrayBase, Axis, Data, Dim, DimMax, Dimension, Ix1, LinalgScalar,
    RemoveAxis, ScalarOperand,
};
use num_traits::{FromPrimitive, One, Zero};
use rand::{distributions::Distribution, Rng};

#[derive(Debug, Copy, Clone)]
pub struct Dense<I> {
    output_size: usize,
    initialiser: I,
}

pub struct DenseSize<I> {
    output_size: usize,
    initialiser: PhantomData<I>,
}

impl<I> Dense<I> {
    #[must_use]
    pub const fn output_size(output_size: usize) -> DenseSize<I> {
        DenseSize {
            output_size,
            initialiser: PhantomData,
        }
    }

    pub fn with_activation<A: Activation>(self, a: A) -> Linear<Self, A> {
        Linear::new(self, a)
    }
}

impl<I> DenseSize<I> {
    pub const fn with_initialiser(self, initialiser: I) -> Dense<I> {
        Dense {
            output_size: self.output_size,
            initialiser,
        }
    }
}

impl<I, F> Graph<F, usize> for Dense<I>
where
    I: Initialiser<F, (usize, usize)>,
{
    type State = DenseState<F>;
    type OutputShape = usize;

    fn get_output_shape(&self) -> usize {
        self.output_size
    }

    fn init_with_random(self, rng: &mut impl Rng, input_size: usize) -> Self::State {
        let d = self
            .initialiser
            .into_distribution((input_size, self.output_size));

        let w = Array2::from_shape_simple_fn((input_size, self.output_size), || d.sample(rng));
        let b = Array1::from_shape_simple_fn(self.output_size, || d.sample(rng));

        DenseState { w, b }
    }
}

#[derive(Debug, Clone)]
pub struct DenseState<F> {
    pub w: Array2<F>,
    pub b: Array1<F>,
}

impl<F, S, D> GraphExec<ArrayBase<S, D>> for DenseState<F>
where
    F: LinalgScalar,
    D: Dimension + DimMax<Ix1, Output = D>,
    S: Data<Elem = F>,
{
    type Output = Array<F, D>;

    fn exec(&self, input: ArrayBase<S, D>) -> Self::Output {
        dot_inner(input, &self.w.view()) + self.b.view()
    }
}

impl<F, D> GraphExecTrain<Array<F, D>> for DenseState<F>
where
    F: LinalgScalar + FromPrimitive + ScalarOperand,
    D: Dimension + DimMax<Ix1, Output = D> + RemoveAxis,
{
    type State = Self::Output;
    fn forward(&self, input: Array<F, D>) -> (Self::State, Self::Output) {
        (input.clone(), self.exec(input))
    }

    fn back(&self, input: Self::State, d_output: Self::Output) -> (Array<F, D>, Self) {
        let di = dot_inner(d_output.clone(), &self.w.t());
        let db = compact_front(d_output.clone()).mean_axis(Axis(0)).unwrap();
        let dw = dot_front(input, d_output);
        (di, Self { w: dw, b: db })
    }
}

impl<T> Mappable<T> for DenseState<T> {
    // not redundant. just forces a capture without needing to clone
    #![allow(clippy::redundant_closure)]

    fn map<F: FnMut(&T) -> T>(&self, mut f: F) -> Self {
        let DenseState { w, b } = self;
        let w = w.map(|a| f(a));
        let b = b.map(f);
        Self { w, b }
    }
    fn map_mut<F: FnMut(&mut T)>(&mut self, mut f: F) {
        self.w.map_mut(|a| f(a));
        self.b.map_mut(f);
    }
    fn map_mut_with<F: FnMut(&mut T, &T)>(&mut self, rhs: &Self, mut f: F) {
        self.w.zip_mut_with(&rhs.w, |a, b| f(a, b));
        self.b.zip_mut_with(&rhs.b, f);
    }
}

impl<T> Shaped<T> for DenseState<T>
where
    T: Clone + Zero + One,
{
    type Shape = Dim<[usize; 2]>;
    fn shape(&self) -> Self::Shape {
        self.w.raw_dim()
    }
    fn zero(shape: Self::Shape) -> Self {
        Self {
            w: Array2::zeros(shape),
            b: Array1::zeros(shape[1]),
        }
    }
    fn one(shape: Self::Shape) -> Self {
        Self {
            w: Array2::ones(shape),
            b: Array1::ones(shape[1]),
        }
    }
    fn iter(shape: Self::Shape, mut i: impl Iterator<Item = T>) -> Self {
        Self {
            w: Array2::from_shape_fn(shape, |_| i.next().unwrap()),
            b: Array1::from_shape_fn(shape[1], |_| i.next().unwrap()),
        }
    }
}

impl<F: H5Type, I> HDF5<F, usize> for Dense<I>
where
    I: Initialiser<F, (usize, usize)>,
{
    fn save(&self, state: &Self::State, group: &hdf5::Group) -> hdf5::Result<()> {
        group
            .new_dataset_builder()
            .with_data(state.w.view())
            .create("weights")?;
        group
            .new_dataset_builder()
            .with_data(state.b.view())
            .create("bias")?;
        Ok(())
    }

    fn load(&self, group: &hdf5::Group) -> hdf5::Result<Self::State> {
        let w = group.dataset("weights")?.read()?;
        let b = group.dataset("bias")?.read()?;

        Ok(DenseState { w, b })
    }
}
