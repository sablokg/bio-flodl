//! Transformer encoder classifier.
//! `cargo run --release --features flodl --example transformer`

mod common;
use bio_flodl::models::transformer::{TransformerClassifier, TransformerConfig};
use bio_flodl::train::{adam, fit, report, TrainConfig};

fn main() -> flodl::Result<()> {
    let (train, val) = common::motif_task(2000, 100, false, 42);
    let model = TransformerClassifier::new(&TransformerConfig {
        alphabet: 4, d_model: 64, heads: 4, layers: 2, d_ff: 128, max_len: 100, dropout: 0.1, classes: 2,
    })?;
    let mut opt = adam(&model, 5e-4);
    fit(&model, &mut opt, &train, Some(&val), &TrainConfig { epochs: 20, batch_size: 64, patience: Some(5), ..Default::default() })?;
    common::print_report("TransformerClassifier", &report(&model, &val, 2, 64)?);
    Ok(())
}
