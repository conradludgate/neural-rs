use ndarray::{Array, Array2, ArrayBase, Data, DataShared, Dimension, Ix2, LinalgScalar, RawData};

pub fn compact_shape(shape: &[usize]) -> (usize, usize) {
    let (last, rest) = shape.split_last().unwrap();
    (rest.iter().product(), *last)
}

pub fn compact_front<S, F, D>(a: ArrayBase<S, D>) -> ArrayBase<S, Ix2>
where
    S: RawData<Elem = F> + Data,
    F: LinalgScalar,
    D: Dimension,
{
    let shape = compact_shape(a.shape());
    a.into_shape(shape).unwrap()
}

/// `AxBxCxI` dot `IxO` -> `AxBxCxO`
/// Converts `AxBxCxI` Array into (AxBxC)xI Array2
/// Performs dot product
/// Then converts back into `AxBxCxO` Array
pub fn dot_inner<S1, S2, F, D>(lhs: ArrayBase<S1, D>, rhs: &ArrayBase<S2, Ix2>) -> Array<F, D>
where
    S1: RawData<Elem = F> + Data,
    S2: RawData<Elem = F> + DataShared,
    F: LinalgScalar,
    D: Dimension,
{
    let mut dim = lhs.raw_dim();
    dim.set_last_elem(rhs.raw_dim().last_elem());

    let i = compact_front(lhs);

    i.dot(rhs).into_shape(dim).unwrap()
}

/// `AxBxCxI` dot `AxBxCxO` -> `IxO`
/// Converts `AxBxCxI` Array into (AxBxC)xI Array2 = l
/// Converts `AxBxCxO` Array into (AxBxC)xO Array2 = r
/// Performs dot product for l.t and r
pub fn dot_front<F, D>(lhs: Array<F, D>, rhs: Array<F, D>) -> Array2<F>
where
    F: LinalgScalar,
    D: Dimension,
{
    let l = compact_front(lhs);
    let r = compact_front(rhs);

    l.t().dot(&r)
}
