use ndarray::LinalgScalar;
use num_traits::{Float, Zero};

use crate::{Mappable, Shaped};

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
    G: Mappable<F> + Clone + Shaped<F>,
{
    pub fn new(alpha: F, beta1: F, beta2: F, epsilon: F, shape: G::Shape) -> Self {
        let zero = G::zero(shape);
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
        // Algorithm defined on Page 2 of https://arxiv.org/pdf/1412.6980v9.pdf
        // https://mlfromscratch.com/optimizers-explained/#actually-explaining-adam

        self.t += 1;

        let b1 = self.beta1;
        let b2 = self.beta2;
        let e = self.epsilon;
        let a = self.alpha;

        let one = F::one();

        // m_t = b1 * m_t-1 + (1 - b1) * g_t
        self.m.map_mut_with(&grads, |m, &g| {
            *m = *m * b1 + g * (one - b1);
        });

        // v_t = b2 * v_t-1 + (1 - b2) * g_t^2
        self.v.map_mut_with(&grads, |v, &g| {
            *v = *v * b2 + g * g * (one - b2);
        });

        // m_t' = m_t / (1 - b1^t)
        let mut mb = self.m.map(|&m| m / (one - b1.powi(self.t)));

        // v_t' = v_t / (1 - b2^t)
        let vb = self.v.map(|&v| v / (one - b2.powi(self.t)));

        // x_t = a * m_t' / (sqrt(v_t') + e)
        mb.map_mut_with(&vb, |m, &v| {
            *m = *m * a / (v.sqrt() + e);
        });

        // g_t = g_t-1 - x_t
        graph.map_mut_with(&mb, |g, &m| {
            *g = *g - m;
        });
    }
}
