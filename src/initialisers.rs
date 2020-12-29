use num_traits::{Float, FromPrimitive};
use rand_distr::{Distribution, Normal, StandardNormal};

pub trait Initialiser<F, S> {
    type Distribution: Distribution<F>;
    fn into_distribution(self, state: S) -> Self::Distribution;
}

#[derive(Debug, Copy, Clone)]
pub struct Xavier;
impl<F> Initialiser<F, (usize, usize)> for Xavier
where
    StandardNormal: Distribution<F>,
    F: Float + FromPrimitive,
{
    type Distribution = Normal<F>;
    fn into_distribution(self, (inputs, _): (usize, usize)) -> Self::Distribution {
        let inputs = F::from_usize(inputs).unwrap();
        let var = F::one() / inputs;
        Normal::new(F::zero(), var.sqrt()).unwrap()
    }
}
