pub mod mse;

pub trait Cost<T> {
    type Inner;
    fn cost(&self, output: &T, expected: &T) -> Self::Inner;
    fn diff(&self, output: &T, expected: &T) -> T;
}

