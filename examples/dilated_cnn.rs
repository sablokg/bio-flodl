//! Residual dilated CNN: exponentially growing receptive field.
//! `cargo run --release --features flodl --example dilated_cnn`

mod common;
use flodl_bio::models::cnn::{DilatedCnnConfig, DilatedResCnn};
use flodl_bio::train::{adam, fit, report, TrainConfig};

fn main() -> flodl::Result<()> {
    let (train, val) = common::motif_task(2000, 200, true, 42);
    // kernel 3, 5 blocks: dilations 1,2,4,8,16 -> receptive field of 2*(1+2+4+8+16)+3 = 65 positions
    let model = DilatedResCnn::new(&DilatedCnnConfig { in_channels: 4, channels: 32, kernel: 3, blocks: 5, dropout: 0.2, classes: 2 })?;
    let mut opt = adam(&model, 1e-3);
    fit(&model, &mut opt, &train, Some(&val), &TrainConfig { epochs: 15, batch_size: 64, patience: Some(4), ..Default::default() })?;
    common::print_report("DilatedResCnn", &report(&model, &val, 2, 64)?);
    Ok(())
}
