use crate::{
    activation::{Activation, Linear},
    array::{compact_front, dot_front, dot_inner},
    initialisers::{Initialiser, Xavier},
    train::GraphExecTrain,
    Graph, GraphExec, Mappable, Shaped,
};
use ndarray::{
    Array, Array1, Array2, ArrayBase, Axis, Data, Dim, Dimension, LinalgScalar, RawData,
    RemoveAxis, ScalarOperand,
};
use num_traits::{FromPrimitive, One, Zero};
use rand::{distributions::Distribution, Rng};

#[derive(Debug, Copy, Clone)]
pub struct Dense<I> {
    output_size: usize,
    initialiser: I,
}

impl Dense<Xavier> {
    pub fn new(output_size: usize) -> Self {
        Dense {
            output_size,
            initialiser: Xavier,
        }
    }
}

impl<I> Dense<I> {
    pub fn with_initialiser<I1>(self, initialiser: I1) -> Dense<I1> {
        Dense {
            output_size: self.output_size,
            initialiser,
        }
    }

    pub fn with_activation<A: Activation<Self>>(self, a: A) -> Linear<Self, A> {
        a.into_activation(self)
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
    D: Dimension,
    S: RawData<Elem = F> + Data,
{
    type Output = Array<F, D>;

    fn exec(&self, input: ArrayBase<S, D>) -> Self::Output {
        dot_inner(input, self.w.view()) + &self.b
    }
}

impl<F, D> GraphExecTrain<Array<F, D>> for DenseState<F>
where
    F: LinalgScalar + FromPrimitive + ScalarOperand,
    D: Dimension + RemoveAxis,
{
    type State = Array<F, D>;
    fn forward(&self, input: Array<F, D>) -> (Self::State, Self::Output) {
        (input.clone(), self.exec(input))
    }

    fn back(&self, input: Self::State, d_output: Self::Output) -> (Array<F, D>, Self) {
        let di = dot_inner(d_output.clone(), self.w.t());
        let db = compact_front(d_output.clone()).mean_axis(Axis(0)).unwrap();
        let dw = dot_front(input, d_output);
        (di, DenseState { w: dw, b: db })
    }
}

impl<T> Mappable<T> for DenseState<T> {
    fn map<F: FnMut(&T) -> T + Clone>(&self, f: F) -> Self {
        let DenseState { w, b } = self;
        let w = w.map(f.clone());
        let b = b.map(f);
        DenseState { w, b }
    }
    fn map_mut<F: FnMut(&mut T) + Clone>(&mut self, f: F) {
        self.w.map_mut(f.clone());
        self.b.map_mut(f);
    }
    fn map_mut_with<F: FnMut(&mut T, &T) + Clone>(&mut self, rhs: &Self, f: F) {
        self.w.zip_mut_with(&rhs.w, f.clone());
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
        DenseState {
            w: Array2::zeros(shape),
            b: Array1::zeros(shape[1]),
        }
    }
    fn one(shape: Self::Shape) -> Self {
        DenseState {
            w: Array2::ones(shape),
            b: Array1::ones(shape[1]),
        }
    }
    fn iter(shape: Self::Shape, mut i: impl Iterator<Item = T>) -> Self {
        DenseState {
            w: Array2::from_shape_fn(shape, |_| i.next().unwrap()),
            b: Array1::from_shape_fn(shape[1], |_| i.next().unwrap()),
        }
    }
}
