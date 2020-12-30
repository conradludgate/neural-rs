use std::ops::{Add, Div, Sub};

use crate::{Cost, GraphExec};

pub trait DerivativeTesting<F> {
    /// Number of adjustable parameters in the graph
    fn len(&self) -> usize;
    /// Adjust a specific parameter by the amount f
    fn get(&self, i: usize) -> F;
    fn set(&mut self, i: usize, f: F);
}

pub fn get_grads<G, C, I, O, F>(graph: &mut G, cost: C, f: F, input: &I, expected: &O) -> G
where
    G: GraphExec<I, Output = O> + DerivativeTesting<F> + Clone,
    C: Cost<O, Inner = F>,
    F: Clone + Add<F, Output = F> + Div<F, Output = F> + Sub<F, Output = F>,
{
    let base_cost = cost.cost(&graph.exec(input), expected);

    let mut grads = graph.clone();

    for i in 0..graph.len() {
        let old = graph.get(i);
        graph.set(i, old.clone() + f.clone());
        let new_cost = cost.cost(&graph.exec(input), expected);
        graph.set(i, old);

        grads.set(i, (new_cost - base_cost.clone()) / f.clone());
    }

    grads
}
