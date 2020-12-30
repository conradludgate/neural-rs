use std::ops::{AddAssign, Mul, Neg};

use super::Optimiser;

#[derive(Debug, Copy, Clone)]
pub struct SGD<F>(F);

impl<F> SGD<F> {
    pub fn new(alpha: F) -> Self {
        SGD(alpha)
    }
}

impl<F, G> Optimiser<G> for SGD<F>
where
    G: Mul<F> + AddAssign<<G as Mul<F>>::Output>,
    F: Clone + Neg<Output=F>,
{
    fn optimise(&mut self, graph: &mut G, grads: G) {
        *graph += grads * -self.0.clone();
    }
}
