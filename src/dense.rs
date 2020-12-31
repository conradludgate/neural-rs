use crate::{
    activation::{Activation, Linear},
    initialisers::{Initialiser, Xavier},
    train::GraphExecTrain,
    Graph, GraphExec, Mappable, Shaped,
};
use ndarray::{Array1, Array2, Axis, Dim, LinalgScalar, ScalarOperand};
use num_traits::{FromPrimitive, Zero, One};
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

        let w = Array2::from_shape_simple_fn((self.output_size, input_size), || d.sample(rng));
        let b = Array1::from_shape_simple_fn(self.output_size, || d.sample(rng));

        DenseState { w, b }
    }
}

#[derive(Debug, Clone)]
pub struct DenseState<F> {
    pub w: Array2<F>,
    pub b: Array1<F>,
}

impl<F> GraphExec<Array1<F>> for DenseState<F>
where
    F: LinalgScalar,
{
    type Output = Array1<F>;

    fn exec(&self, input: &Array1<F>) -> Self::Output {
        self.w.dot(input) + self.b.clone()
    }
}

impl<F> GraphExec<Array2<F>> for DenseState<F>
where
    F: LinalgScalar,
{
    type Output = Array2<F>;

    fn exec(&self, input: &Array2<F>) -> Self::Output {
        self.w.dot(input) + self.b.clone().insert_axis(Axis(1))
    }
}

impl<F> GraphExecTrain<Array1<F>> for DenseState<F>
where
    F: LinalgScalar,
{
    type State = Array1<F>;
    fn forward(&self, input: &Array1<F>) -> (Self::State, Self::Output) {
        (input.clone(), self.exec(input))
    }

    fn back(&self, input: Self::State, d_output: Self::Output) -> (Array1<F>, Self) {
        let di = self.w.t().dot(&d_output);
        let db = d_output.clone();
        let dw = d_output
            .insert_axis(Axis(1))
            .dot(&input.insert_axis(Axis(0)));
        (di, DenseState { w: dw, b: db })
    }
}

impl<F> GraphExecTrain<Array2<F>> for DenseState<F>
where
    F: LinalgScalar + FromPrimitive + ScalarOperand,
{
    type State = Array2<F>;
    fn forward(&self, input: &Array2<F>) -> (Self::State, Self::Output) {
        (input.clone(), self.exec(input))
    }

    fn back(&self, input: Self::State, d_output: Self::Output) -> (Array2<F>, Self) {
        let batch_size = F::from_usize(d_output.raw_dim()[1]).unwrap();
        let di = self.w.t().dot(&d_output);
        let db = d_output.mean_axis(Axis(1)).unwrap();
        let dw = d_output.dot(&input.t()) / batch_size;
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
            b: Array1::zeros(shape[0]),
        }
    }
    fn one(shape: Self::Shape) -> Self {
        DenseState {
            w: Array2::ones(shape),
            b: Array1::ones(shape[0]),
        }
    }
    fn iter(shape: Self::Shape, mut i: impl Iterator<Item=T>) -> Self {
        DenseState {
            w: Array2::from_shape_fn(shape, |_| i.next().unwrap()),
            b: Array1::from_shape_fn(shape[0], |_| i.next().unwrap()),
        }
    }
}
