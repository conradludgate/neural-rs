pub mod adam;
pub mod sgd;

pub trait Optimiser<G> {
    fn optimise(&mut self, graph: &mut G, grads: G);
}
