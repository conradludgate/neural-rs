use crate::{train::GraphExecTrain, Graph, GraphExec, Mappable, Shaped};
use rand::Rng;

impl<I, G0, G1, F> Graph<F, I> for (G0, G1)
where
    G0: Graph<F, I>,
    G1: Graph<F, G0::OutputShape>,
{
    type State = (G0::State, G1::State);
    type OutputShape = G1::OutputShape;

    fn get_output_shape(&self) -> Self::OutputShape {
        self.1.get_output_shape()
    }

    fn init_with_random(self, rng: &mut impl Rng, input_shape: I) -> Self::State {
        let s0 = self.0.get_output_shape();
        (
            self.0.init_with_random(rng, input_shape),
            self.1.init_with_random(rng, s0),
        )
    }
}

impl<G0, G1, Input> GraphExec<Input> for (G0, G1)
where
    G0: GraphExec<Input>,
    G1: GraphExec<G0::Output>,
{
    type Output = G1::Output;
    fn exec(&self, input: Input) -> Self::Output {
        let input = self.0.exec(input);
        self.1.exec(input)
    }
}

impl<G0, G1, Input> GraphExecTrain<Input> for (G0, G1)
where
    G0: GraphExecTrain<Input>,
    G1: GraphExecTrain<G0::Output>,
{
    type State = (G0::State, G1::State);
    fn forward(&self, input: Input) -> (Self::State, Self::Output) {
        let (s0, input) = self.0.forward(input);
        let (s1, input) = self.1.forward(input);
        ((s0, s1), input)
    }

    fn back(&self, state: Self::State, d_output: Self::Output) -> (Input, Self) {
        let (s0, s1) = state;
        let (d_output, d1) = self.1.back(s1, d_output);
        let (d_output, d0) = self.0.back(s0, d_output);
        (d_output, (d0, d1))
    }
}

impl<S, T, U> Mappable<S> for (T, U)
where
    T: Mappable<S>,
    U: Mappable<S>,
{
    fn map<F: FnMut(&S) -> S + Clone>(&self, f: F) -> Self {
        let t = self.0.map(f.clone());
        let u = self.1.map(f);
        (t, u)
    }
    fn map_mut<F: FnMut(&mut S) + Clone>(&mut self, f: F) {
        self.0.map_mut(f.clone());
        self.1.map_mut(f);
    }
    fn map_mut_with<F: FnMut(&mut S, &S) + Clone>(&mut self, rhs: &Self, f: F) {
        self.0.map_mut_with(&rhs.0, f.clone());
        self.1.map_mut_with(&rhs.1, f);
    }
}

impl<F, T, U> Shaped<F> for (T, U)
where
    T: Shaped<F>,
    U: Shaped<F>,
{
    type Shape = (T::Shape, U::Shape);
    fn shape(&self) -> Self::Shape {
        (self.0.shape(), self.1.shape())
    }
    fn zero(shape: Self::Shape) -> Self {
        (T::zero(shape.0), U::zero(shape.1))
    }
    fn one(shape: Self::Shape) -> Self {
        (T::one(shape.0), U::one(shape.1))
    }
    fn iter(shape: Self::Shape, mut i: impl Iterator<Item = F>) -> Self {
        (T::iter(shape.0, &mut i), U::iter(shape.1, &mut i))
    }
}

/// Converts the provided values into a nested chain of tuples.
/// Works by taking each pair of expressions, converting them into a tuple,
/// Then pushing all of them into the macro recursively
///
/// ```
/// use linear_networks::net;
///
/// // These two expressions are the same
/// let a = net!(0, 1, 2, 3);
/// let b = net!((0, 1), (2, 3));
/// assert_eq!(a, b);
/// ```
///
/// There's an edge case to handle odd numbered inputs.
/// It leaves the first input and pairs up the rest of them
///
/// ```
/// use linear_networks::net;
///
/// let a = net!(0, 1, 2, 3, 4);
/// let b = net!(0, (1, 2), (3, 4));
/// let c = net!(0, ((1, 2), (3, 4)));
/// assert_eq!(a, b);
/// assert_eq!(a, c);
/// ```
#[macro_export]
macro_rules! net {
    ($g0:expr) => {
        $g0
    };
    ($($g0:expr, $g1:expr),*) => {
        $crate::net!($(
            ($g0, $g1)
        ),*)
    };
    ($g:expr, $($g0:expr, $g1:expr),*) => {
        $crate::net!(
            $g,
            $(
                ($g0, $g1)
            ),*
        )
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_tuple_macro() {
        // single value
        let t = net!(0);
        assert_eq!(t, 0);

        // two values
        let t = net!(0, 1);
        assert_eq!(t, (0, 1));

        // 8 values (balanced nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5, 6, 7);
        assert_eq!(t, (((0, 1), (2, 3)), ((4, 5), (6, 7))));

        // 7 values (off balance nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5, 6);
        assert_eq!(t, ((0, (1, 2)), ((3, 4), (5, 6))));

        // 6 values (off balance nested binary tree)
        let t = net!(0, 1, 2, 3, 4, 5);
        assert_eq!(t, ((0, 1), ((2, 3), (4, 5))));
    }
}
