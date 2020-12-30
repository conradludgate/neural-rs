use ndarray::LinalgScalar;

use crate::Mappable;

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
    G: Mappable<F>,
    F: LinalgScalar,
{
    fn optimise(&mut self, graph: &mut G, grads: G) {
        graph.map_mut_with(&grads, |theta, &g| *theta = *theta - g * self.0);
    }
}
