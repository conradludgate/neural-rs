mod parse;

use std::mem::MaybeUninit;

use indicatif::{ProgressBar, ProgressStyle};
use linear_networks::{
    activation::{relu::Relu, sigmoid::Sigmoid},
    cost::MSE,
    dense::Dense,
    initialisers::Xavier,
    optimise::{sgd::SGD, Train},
    tuple, Graph, GraphExec,
};
use ndarray::Array2;
use rand::prelude::*;

fn main() {
    // Load MNIST data set
    let data = parse::load_data();

    // Create a new compute graph which uses three Dense components
    // With the input having size 28*28 and the output having size 10
    // Initialise it with uniform random data
    let network: _ = tuple![
        Dense::new(16, Xavier).with_activation(Relu),
        Dense::new(16, Xavier).with_activation(Relu),
        Dense::new(10, Xavier).with_activation(Sigmoid)
    ]
    .input_shape(28 * 28);

    // New trainer with mean squared error cost function and
    // stochastic gradient descent optimisation (alpha=0.1)
    let mut trainer: _ = Train::new(network, MSE, SGD::new(0.1));

    let mut costs = vec![];

    const BATCH_SIZE: usize = 120;
    let total_batches = ((data.training.images.len() + BATCH_SIZE - 1) / BATCH_SIZE) as u64;

    let pb = ProgressBar::new(total_batches)
        .with_style(ProgressStyle::default_bar().template("{msg}\n{wide_bar} {pos}/{len}"));
    pb.set_message(&format!("Training step {}", 1));

    for _ in 0..20 {
        let batches = make_bacthes(data.training.images.len(), BATCH_SIZE);
        let len = batches.len() as f32;
        let mut total_cost = 0.0;

        for batch in batches {
            let (input, expected) = process_batch(&data.training, &batch);
            total_cost += trainer.train(&input, &expected);

            pb.inc(1);
        }

        costs.push(total_cost / len);

        pb.reset();
        pb.set_message(&format!(
            "Training step {}. Previous cost: {:?}",
            costs.len() + 1,
            total_cost / len
        ));
    }

    let network = trainer.into_inner();

    println!("network: {:?}", network);

    let (input, expected) = process_batch(&data.testing, &vec![0]);

    let output = network.exec(&input);
    println!("output: {:?}", output);
    println!("expected: {:?}", expected);
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

fn process_batch(data: &parse::DataSet, batch: &Vec<usize>) -> (Array2<f32>, Array2<f32>) {
    unsafe {
        let mut input = Array2::maybe_uninit((28 * 28, batch.len()));
        let mut expected = Array2::maybe_uninit((10, batch.len()));

        for (i, &j) in batch.iter().enumerate() {
            let image = &data.images[j];
            for (j, &b) in image.iter().enumerate() {
                *input[(j, i)].as_mut_ptr() = (b as f32) / 255.0;
            }
            expected.column_mut(i).fill(MaybeUninit::new(0.0));
            *expected[(data.labels[j] as usize, i)].as_mut_ptr() = 1.0;
        }

        (input.assume_init(), expected.assume_init())
    }
}
