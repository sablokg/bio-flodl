//! Bidirectional LSTM and GRU classifiers.
//! `cargo run --release --features flodl --example birnn`

mod common;
use flodl_bio::models::rnn::{BiRnnClassifier, RnnKind};
use flodl_bio::train::{adam, fit, report, TrainConfig};

fn main() -> flodl::Result<()> {
    let (train, val) = common::motif_task(2000, 60, false, 42); // [L, 4] layout
    for (name, kind) in [("BiLSTM", RnnKind::Lstm), ("BiGRU", RnnKind::Gru)] {
        let model = BiRnnClassifier::new(kind, 4, 32, 1, 0.2, 2)?;
        let mut opt = adam(&model, 2e-3);
        fit(&model, &mut opt, &train, Some(&val), &TrainConfig { epochs: 15, batch_size: 64, patience: Some(4), verbose: false, ..Default::default() })?;
        common::print_report(name, &report(&model, &val, 2, 64)?);
    }
    Ok(())
}
