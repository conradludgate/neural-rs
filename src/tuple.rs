use std::ops::{Add, AddAssign, Mul};

use crate::{derivative::DerivativeTesting, Graph, GraphExec, GraphExecTrain};
use ndarray::ScalarOperand;
use rand::Rng;

/// Tuple2<T, U> is essentially (T, U) but redefined to allow implementing more traits on it.
#[derive(Debug, Copy, Clone)]
pub struct Tuple2<T, U>(T, U);

impl<T, U> Tuple2<T, U> {
    /// Create a new Tuple2
    pub fn new(t: T, u: U) -> Self {
        Tuple2(t, u)
    }
}

impl<I, G0, G1, F> Graph<F, I> for Tuple2<G0, G1>
where
    G0: Graph<F, I>,
    G1: Graph<F, G0::OutputShape>,
{
    type State = Tuple2<G0::State, G1::State>;
    type OutputShape = G1::OutputShape;

    fn get_output_shape(&self) -> Self::OutputShape {
        self.1.get_output_shape()
    }

    fn init_with_random(self, rng: &mut impl Rng, input_shape: I) -> Self::State {
        let s0 = self.0.get_output_shape();
        Tuple2(
            self.0.init_with_random(rng, input_shape),
            self.1.init_with_random(rng, s0),
        )
    }
}

impl<G0, G1, Input> GraphExec<Input> for Tuple2<G0, G1>
where
    G0: GraphExec<Input>,
    G1: GraphExec<G0::Output>,
{
    type Output = G1::Output;
    fn exec(&self, input: &Input) -> Self::Output {
        let input = self.0.exec(&input);
        self.1.exec(&input)
    }
}

impl<G0, G1, Input> GraphExecTrain<Input> for Tuple2<G0, G1>
where
    G0: GraphExecTrain<Input>,
    G1: GraphExecTrain<G0::Output>,
{
    type State = Tuple2<G0::State, G1::State>;
    fn forward(&self, input: &Input) -> (Self::State, Self::Output) {
        let (s0, input): _ = self.0.forward(&input);
        let (s1, input): _ = self.1.forward(&input);
        (Tuple2(s0, s1), input)
    }

    fn back(&self, state: Self::State, d_output: Self::Output) -> (Input, Self) {
        let Tuple2(s0, s1): _ = state;
        let (d_output, d1): _ = self.1.back(s1, d_output);
        let (d_output, d0): _ = self.0.back(s0, d_output);
        (d_output, Tuple2(d0, d1))
    }
}

/// Converts the provided values into a nested chain of [Tuple2]s.
/// Works by taking each pair of expressions, converting them into a [Tuple2],
/// Then pushing all of them into the macro recursively
///
/// ```
/// # use linear_networks::tuple::Tuple2;
/// use linear_networks::tuple;
///
/// // These two expressions are the same
/// let a = tuple!(0, 1, 2, 3);
/// let b = tuple!(Tuple2::new(0, 1), Tuple2::new(2, 3));
/// assert_eq!(a, b);
/// ```
///
/// There's an edge case to handle odd numbered inputs.
/// It leaves the first input and pairs up the rest of them
///
/// ```
/// # use linear_networks::tuple::Tuple2;
/// use linear_networks::tuple;
///
/// let a = tuple!(0, 1, 2, 3, 4);
/// let b = tuple!(0, Tuple2::new(1, 2), Tuple2::new(3, 4));
/// let c = tuple!(0, Tuple2::new(Tuple2::new(1, 2), Tuple2::new(3, 4)));
/// assert_eq!(a, b);
/// assert_eq!(a, c);
/// ```
#[macro_export]
macro_rules! tuple {
    ($g0:expr) => {
        $g0
    };
    ($($g0:expr, $g1:expr),*) => {
        tuple!($(
            $crate::tuple::Tuple2::new($g0, $g1)
        ),*);
    };
    ($g:expr, $($g0:expr, $g1:expr),*) => {
        tuple!(
            $g,
            $(
                $crate::tuple::Tuple2::new($g0, $g1)
            ),*
        );
    };
}

impl<T1, U1, T2, U2> Add<Tuple2<T2, U2>> for Tuple2<T1, U1>
where
    T1: Add<T2>,
    U1: Add<U2>,
{
    type Output = Tuple2<T1::Output, U1::Output>;
    fn add(self, rhs: Tuple2<T2, U2>) -> Self::Output {
        Tuple2(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl<T1, U1, T2, U2> AddAssign<Tuple2<T2, U2>> for Tuple2<T1, U1>
where
    T1: AddAssign<T2>,
    U1: AddAssign<U2>,
{
    fn add_assign(&mut self, rhs: Tuple2<T2, U2>) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}
impl<T1, U1, T2, U2> Mul<Tuple2<T2, U2>> for Tuple2<T1, U1>
where
    T1: Mul<T2>,
    U1: Mul<U2>,
{
    type Output = Tuple2<T1::Output, U1::Output>;
    fn mul(self, rhs: Tuple2<T2, U2>) -> Self::Output {
        Tuple2(self.0 * rhs.0, self.1 * rhs.1)
    }
}
impl<T, U, S> Mul<S> for Tuple2<T, U>
where
    T: Mul<S>,
    U: Mul<S>,
    S: ScalarOperand + Clone,
{
    type Output = Tuple2<T::Output, U::Output>;
    fn mul(self, rhs: S) -> Self::Output {
        Tuple2(self.0 * rhs.clone(), self.1 * rhs)
    }
}
impl<T1, U1, T2, U2> PartialEq<Tuple2<T2, U2>> for Tuple2<T1, U1>
where
    T1: PartialEq<T2>,
    U1: PartialEq<U2>,
{
    fn eq(&self, rhs: &Tuple2<T2, U2>) -> bool {
        self.0 == rhs.0 && self.1 == rhs.1
    }
}

impl<F, T, U> DerivativeTesting<F> for Tuple2<T, U>
where
    T: DerivativeTesting<F>,
    U: DerivativeTesting<F>,
{
    fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }
    fn get(&self, i: usize) -> F {
        let l = self.0.len();
        if i < l {
            self.0.get(i)
        } else {
            self.1.get(i - l)
        }
    }

    fn set(&mut self, i: usize, f: F) {
        let l = self.0.len();
        if i < l {
            self.0.set(i, f);
        } else {
            self.1.set(i - l, f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tuple2;
    #[test]
    fn test_tuple_macro() {
        // single value
        let t = tuple!(0);
        assert_eq!(t, 0);

        // two values
        let t = tuple!(0, 1);
        assert_eq!(t, Tuple2::new(0, 1));

        // 8 values (balanced nested binary tree)
        let t = tuple!(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq!(
            t,
            Tuple2::new(
                Tuple2::new(Tuple2::new(0, 1), Tuple2::new(2, 3)),
                Tuple2::new(Tuple2::new(4, 5), Tuple2::new(6, 7))
            )
        );

        // 7 values (off balance nested binary tree)
        let t = tuple!(0, 1, 2, 3, 4, 5, 6);
        assert_eq!(
            t,
            Tuple2::new(
                Tuple2::new(0, Tuple2::new(1, 2)),
                Tuple2::new(Tuple2::new(3, 4), Tuple2::new(5, 6))
            )
        );

        // 6 values (off balance nested binary tree)
        let t = tuple!(0, 1, 2, 3, 4, 5);
        assert_eq!(
            t,
            Tuple2::new(
                Tuple2::new(0, 1),
                Tuple2::new(Tuple2::new(2, 3), Tuple2::new(4, 5))
            )
        );
    }
}
