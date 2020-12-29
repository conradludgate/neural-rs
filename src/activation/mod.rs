pub mod sigmoid;

pub trait Activation<G> {
    type Activation;
    fn into_activation(self, g: G) -> Self::Activation;
}
