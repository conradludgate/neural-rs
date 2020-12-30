use ndarray::LinalgScalar;
use num_traits::{Float, Zero};

use crate::Mappable;

use super::Optimiser;

#[derive(Debug, Copy, Clone)]
pub struct Adam<F, G> {
    alpha: F,
    beta1: F,
    beta2: F,
    epsilon: F,
    m: G,
    v: G,
    t: i32,
}

impl<F, G> Adam<F, G>
where
    F: Zero + Copy,
    G: Mappable<F> + Clone,
{
    pub fn new(alpha: F, beta1: F, beta2: F, epsilon: F, graph: &G) -> Self {
        let zero = F::zero();
        let zero = graph.map(|_| zero);
        Adam {
            alpha,
            beta1,
            beta2,
            epsilon,
            m: zero.clone(),
            v: zero,
            t: 0,
        }
    }
}

impl<F, G> Optimiser<G> for Adam<F, G>
where
    G: Mappable<F>,
    F: LinalgScalar + Float,
{
    fn optimise(&mut self, graph: &mut G, grads: G) {
        self.t += 1;

        let b1 = self.beta1;
        let b2 = self.beta2;
        let e = self.epsilon;
        let a = self.alpha;

        let one = F::one();

        self.m.map_mut_with(&grads, |m, &g| {
            *m = *m * b1 + g * (one - b1);
        });

        self.v.map_mut_with(&grads, |v, &g| {
            *v = *v * b2 + g * g * (one - b2);
        });

        let mut mb = self.m.map(|&m| m / (one - b1.powi(self.t)));
        let vb = self.v.map(|&v| v / (one - b2.powi(self.t)));

        mb.map_mut_with(&vb, |m, &v| {
            *m = *m * a / (v.sqrt() + e);
        });

        graph.map_mut_with(&mb, |g, &m| {
            *g = *g + m;
        });
    }
}
