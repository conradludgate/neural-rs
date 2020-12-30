use crate::{Graph, GraphExec, GraphExecTrain, Mappable, activation::{Activation, Linear}, initialisers::Initialiser};
use ndarray::{Array1, Array2, Axis, LinalgScalar, ScalarOperand};
use num_traits::{FromPrimitive};
use rand::{distributions::Distribution, Rng};

#[derive(Debug, Copy, Clone)]
pub struct Dense<I> {
    output_size: usize,
    initialiser: I,
}

impl<I> Dense<I> {
    pub fn new(output_size: usize, initialiser: I) -> Self {
        Dense {
            output_size,
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

// impl<F> Add<DenseState<F>> for DenseState<F>
// where
//     F: LinalgScalar,
// {
//     type Output = DenseState<F>;
//     fn add(self, rhs: DenseState<F>) -> DenseState<F> {
//         DenseState {
//             w: self.w + rhs.w,
//             b: self.b + rhs.b,
//         }
//     }
// }
// impl<F> AddAssign<DenseState<F>> for DenseState<F>
// where
//     F: LinalgScalar + AddAssign<F>,
// {
//     fn add_assign(&mut self, rhs: DenseState<F>) {
//         self.w += &rhs.w;
//         self.b += &rhs.b;
//     }
// }
// impl<F> Mul<F> for DenseState<F>
// where
//     F: LinalgScalar + ScalarOperand,
// {
//     type Output = DenseState<F>;
//     fn mul(self, rhs: F) -> DenseState<F> {
//         DenseState {
//             w: self.w * rhs,
//             b: self.b * rhs,
//         }
//     }
// }

// impl<F> DerivativeTesting<F> for DenseState<F>
// where
//     F: LinalgScalar,
// {
//     fn len(&self) -> usize {
//         self.w.len() + self.b.len()
//     }

//     fn get(&self, i: usize) -> F {
//         let l = self.w.len();
//         if i < l {
//             self.w.as_slice().unwrap()[i]
//         } else {
//             self.b[i - l]
//         }
//     }

//     fn set(&mut self, i: usize, f: F) {
//         let l = self.w.len();
//         if i < l {
//             self.w.as_slice_mut().unwrap()[i] = f;
//         } else {
//             self.b[i - l] = f;
//         }
//     }
// }


impl<T> Mappable<T> for DenseState<T> {
    fn map<F: FnMut(&T) -> T + Clone>(&self, f: F) -> Self {
        let DenseState { w, b } = self;
        let w = w.map(f.clone());
        let b = b.map(f);
        DenseState{w, b}
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
