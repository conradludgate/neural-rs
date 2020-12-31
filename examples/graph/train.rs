use linear_networks::{
    activation::{relu::Relu, sigmoid::Sigmoid},
    cost::mse::MSE,
    dense::Dense,
    initialisers::Xavier,
    optimise::adam::Adam,
    train::Train,
    tuple, Graph, Shaped,
};
use ndarray::Array2;
use std::sync::mpsc;

use crate::{event::Event, parse};

pub fn train(tx: mpsc::Sender<Event>) {
    // Load MNIST data set
    let data = parse::load_data();
    let training_data = process_data(&data.training);

    // Create a new compute graph which uses three Dense components
    // With the input having size 28*28 and the output having size 10
    // Initialise it with uniform random data
    let network: _ = tuple![
        Dense::new(16)
            .with_initialiser(Xavier)
            .with_activation(Relu),
        Dense::new(16)
            .with_initialiser(Xavier)
            .with_activation(Relu),
        Dense::new(10)
            .with_initialiser(Xavier)
            .with_activation(Sigmoid)
    ]
    .input_shape(28 * 28);

    // New trainer with mean squared error cost function
    let optimiser: _ = Adam::new(0.001, 0.9, 0.99, 1e-8, network.shape());
    let mut trainer: _ = Train {
        graph: network,
        optimiser,
        cost: MSE,
        regularisation: None,
        dropout: 0.0,
    };

    const BATCH_SIZE: usize = 120;

    for _ in 0.. {
        let cost =
            trainer.perform_epoch(training_data.0.view(), training_data.1.view(), BATCH_SIZE);
        tx.send(Event::EpochComplete(cost)).unwrap();
    }
}

fn process_data(data: &parse::DataSet) -> (Array2<f64>, Array2<f64>) {
    unsafe {
        let data_len = data.images.len();
        assert_eq!(data_len, data.labels.len());

        let mut input = Array2::uninitialized((data_len, 28 * 28));
        let mut expected = Array2::zeros((data_len, 10));

        for (i, image) in data.images.iter().enumerate() {
            for (j, &b) in image.iter().enumerate() {
                input[(i, j)] = (b as f64) / 255.0;
            }
        }
        for (i, &label) in data.labels.iter().enumerate() {
            expected[(i, label as usize)] = 1.0;
        }

        (input, expected)
    }
}
