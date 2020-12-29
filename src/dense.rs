use std::ops::{Add, AddAssign, Mul};

use crate::{Graph, GraphExec, GraphExecTrain, activation::Activation, initialisers::Initialiser};
use ndarray::{Array1, Array2, Axis, LinalgScalar, ScalarOperand};
use num_traits::FromPrimitive;
use rand::{distributions::Distribution, Rng};

#[derive(Debug, Copy, Clone)]
pub struct Dense<I> {
    input_size: usize,
    initialiser: I,
}

impl<I> Dense<I> {
    pub fn new(input_size: usize, initialiser: I) -> Self {
        Dense {
            input_size,
            initialiser,
        }
    }
    pub fn with_activation<A: Activation<Self>>(self, a: A) -> A::Activation {
        a.into_activation(self)
    }
}

impl<I, F> Graph<F> for Dense<I>
where
    I: Initialiser<F, (usize, usize)>,
{
    type State = DenseState<F>;

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn init_with_random(self, rng: &mut impl Rng, output_size: usize) -> Self::State {
        let d = self
            .initialiser
            .into_distribution((self.input_size, output_size));

        let w = Array2::from_shape_simple_fn((output_size, self.input_size), || d.sample(rng));
        let b = Array1::from_shape_simple_fn(output_size, || d.sample(rng));

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

    fn back(&self, state: Self::State, d_output: Self::Output) -> (Array1<F>, Self) {
        let di = self.w.t().dot(&d_output);
        let db = d_output.clone();
        let dw = d_output
            .insert_axis(Axis(1))
            .dot(&state.insert_axis(Axis(0)));
        (di, DenseState { w: dw, b: db })
    }
}

impl<F> GraphExecTrain<Array2<F>> for DenseState<F>
where
    F: LinalgScalar + FromPrimitive,
{
    type State = Array2<F>;
    fn forward(&self, input: &Array2<F>) -> (Self::State, Self::Output) {
        (input.clone(), self.exec(input))
    }

    fn back(&self, state: Self::State, d_output: Self::Output) -> (Array2<F>, Self) {
        let di = self.w.t().dot(&d_output);
        let db = d_output.mean_axis(Axis(1)).unwrap();
        let dw = d_output.dot(&state.t());
        (di, DenseState { w: dw, b: db })
    }
}

impl<F> Add<DenseState<F>> for DenseState<F>
where
    F: LinalgScalar,
{
    type Output = DenseState<F>;
    fn add(self, rhs: DenseState<F>) -> DenseState<F> {
        DenseState {
            w: self.w + rhs.w,
            b: self.b + rhs.b,
        }
    }
}
impl<F> AddAssign<DenseState<F>> for DenseState<F>
where
    F: LinalgScalar + AddAssign<F>,
{
    fn add_assign(&mut self, rhs: DenseState<F>) {
        self.w += &rhs.w;
        self.b += &rhs.b;
    }
}
impl<F> Mul<F> for DenseState<F>
where
    F: LinalgScalar + ScalarOperand,
{
    type Output = DenseState<F>;
    fn mul(self, rhs: F) -> DenseState<F> {
        DenseState {
            w: self.w + rhs,
            b: self.b + rhs,
        }
    }
}
