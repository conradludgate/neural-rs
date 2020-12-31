use std::{mem::MaybeUninit};

use linear_networks::{Graph, Shaped, activation::{relu::Relu, sigmoid::Sigmoid}, cost::mse::MSE, dense::Dense, initialisers::Xavier, optimise::{adam::Adam}, train::{Regularisation, Train}, tuple};
use ndarray::Array2;
use rand::prelude::*;
use std::sync::mpsc;

use crate::{event::Event, parse};

pub fn train(tx: mpsc::Sender<Event>) {
    // Load MNIST data set
    let data = parse::load_data();

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
    let mut trainer: _ = Train{
        graph: network,
        optimiser,
        cost: MSE,
        regularisation: None,
        dropout: 0.2,
    };

    const BATCH_SIZE: usize = 120;

    for _ in 0.. {
        let batches = make_bacthes(data.training.images.len(), BATCH_SIZE);
        let n = batches.len();
        let len = n as f64;
        let mut total_cost = 0.0;

        for (i, batch) in batches.into_iter().enumerate() {
            let (input, expected) = process_batch(&data.training, &batch);
            total_cost += trainer.train(&input, &expected);

            tx.send(Event::Step(i, n)).unwrap();
        }
        tx.send(Event::EpochComplete(total_cost / len)).unwrap();
    }
}

fn make_bacthes(length: usize, batch_size: usize) -> Vec<Vec<usize>> {
    let mut numbers = (0..length).collect::<Vec<_>>();
    let mut rng = thread_rng();
    numbers.shuffle(&mut rng);
    let mut batches = vec![];
    for i in (0..=length - batch_size).step_by(batch_size) {
        batches.push(numbers[i..i + batch_size].to_vec());
    }
    if length % batch_size != 0 {
        batches.push(numbers[length - batch_size..].to_vec());
    }

    batches
}

fn process_batch(data: &parse::DataSet, batch: &Vec<usize>) -> (Array2<f64>, Array2<f64>) {
    unsafe {
        let mut input = Array2::maybe_uninit((28 * 28, batch.len()));
        let mut expected = Array2::maybe_uninit((10, batch.len()));

        for (i, &j) in batch.iter().enumerate() {
            let image = &data.images[j];
            for (j, &b) in image.iter().enumerate() {
                *input[(j, i)].as_mut_ptr() = (b as f64) / 255.0;
            }
            expected.column_mut(i).fill(MaybeUninit::new(0.0));
            *expected[(data.labels[j] as usize, i)].as_mut_ptr() = 1.0;
        }

        (input.assume_init(), expected.assume_init())
    }
}
