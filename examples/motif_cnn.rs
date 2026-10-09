//! Multi-width motif CNN on a planted TATA-box-like motif.
//! `cargo run --release --features flodl --example motif_cnn` (needs libtorch)

mod common;
use bio_flodl::models::cnn::{MotifCnn, MotifCnnConfig};
use bio_flodl::train::{adam, fit, report, TrainConfig};

fn main() -> flodl::Result<()> {
    let (train, val) = common::motif_task(2000, 100, true, 42);
    let model = MotifCnn::new(&MotifCnnConfig { in_channels: 4, filters: 32, kernel_sizes: vec![6, 9, 12], dropout: 0.2, classes: 2 })?;
    let mut opt = adam(&model, 1e-3);
    fit(&model, &mut opt, &train, Some(&val), &TrainConfig { epochs: 15, batch_size: 64, patience: Some(4), ..Default::default() })?;
    common::print_report("MotifCnn", &report(&model, &val, 2, 64)?);
    Ok(())
}
