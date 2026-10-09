//! Build a custom model from flodl layers with the composition containers, then train it.
//! Uses only constructors verified against flodl's docs (Conv1d, Dropout, Linear).
//! To add pooling, batch-norm, Conv2d, etc., construct them with flodl's own constructors
//! (check docs.rs for their argument lists) and `.push(...)` them in.
//! `cargo run --release --features flodl --example compose_sequential`

mod common;
use bio_flodl::prelude::*;

fn main() -> flodl::Result<()> {
    let (train_set, val) = common::motif_task(2000, 100, true, 42);

    // two parallel motif scanners (widths 6 and 12), each: conv -> relu -> global max pool
    let scanner = |k: i64| -> flodl::Result<Sequential> {
        Ok(Sequential::new().push(Conv1d::new(4, 32, k)?).push(Lambda::relu()).push(Lambda::global_max_pool1d()))
    };
    let model = Sequential::new()
        .push(ConcatBranches::new(1).branch(scanner(6)?).branch(scanner(12)?)) // [B, 64]
        .push(Dropout::new(0.2))
        .push(Linear::new(64, 2)?);

    let mut optimizer = adam(&model, 1e-3);
    fit(&model, &mut optimizer, &train_set, Some(&val), &TrainConfig { epochs: 12, batch_size: 64, ..Default::default() })?;
    common::print_report("Sequential(ConcatBranches)", &report(&model, &val, 2, 64)?);
    Ok(())
}
