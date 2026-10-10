//! Model tier, feature `flodl`: `cargo test --features flodl`.
//! The GPU test is ignored by default; on a CUDA machine run
//! `cargo test --features cuda --test models -- --ignored`.
#![cfg(feature = "flodl")]

use flodl_bio::alphabet::Alphabet;
use flodl_bio::data::{synthetic_motif, Dataset};
use flodl_bio::prelude::*;

fn motif_data(channels_first: bool) -> (Dataset, Dataset) {
    let (seqs, labels) = synthetic_motif(200, 50, b"TATAAA", 7);
    let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
    Dataset::from_sequences(&refs, &labels, &Alphabet::DNA, 50, channels_first)
        .unwrap()
        .split(0.2, 1)
}

/// Moves `model` to `dev`, checks that its state followed, then trains and evaluates it there.
fn train_on(model: &dyn Module, dev: Device, train: &Dataset, val: &Dataset) -> Result<()> {
    model.move_to_device(dev);
    assert!(model
        .parameters()
        .iter()
        .all(|p| p.variable.device() == dev));
    assert!(model.buffers().iter().all(|b| b.device() == dev));
    let cfg = TrainConfig {
        epochs: 1,
        verbose: false,
        ..Default::default()
    };
    fit(model, &mut adam(model, 1e-3), train, Some(val), &cfg)?;
    report(model, val, 2, 32)?;
    Ok(())
}

/// Trains every bundled model, plus a container holding BatchNorm, on `dev`.
fn train_every_model(dev: Device) -> Result<()> {
    let (train, val) = motif_data(true);
    let composed = Sequential::new()
        .push(Conv1d::new(4, 16, 8)?)
        .push(Lambda::relu())
        .push(Lambda::global_max_pool1d())
        .push(BatchNorm::new(16)?)
        .push(Linear::new(16, 2)?);
    let motif = MotifCnn::new(&MotifCnnConfig {
        in_channels: 4,
        filters: 8,
        kernel_sizes: vec![6, 9],
        dropout: 0.0,
        classes: 2,
    })?;
    let dilated = DilatedResCnn::new(&DilatedCnnConfig {
        in_channels: 4,
        channels: 8,
        kernel: 3,
        blocks: 2,
        dropout: 0.0,
        classes: 2,
    })?;
    for model in [&composed as &dyn Module, &motif, &dilated] {
        train_on(model, dev, &train, &val)?;
    }

    let (train, val) = motif_data(false);
    let transformer = TransformerClassifier::new(&TransformerConfig {
        alphabet: 4,
        d_model: 16,
        heads: 2,
        layers: 1,
        d_ff: 32,
        max_len: 50,
        dropout: 0.0,
        classes: 2,
    })?;
    let lstm = BiRnnClassifier::new(RnnKind::Lstm, 4, 8, 1, 0.0, 2)?;
    let gru = BiRnnClassifier::new(RnnKind::Gru, 4, 8, 1, 0.0, 2)?;
    let mlp = FlatMlp::new(50 * 4, &[16], 2, 0.0)?;
    for model in [&transformer as &dyn Module, &lstm, &gru, &mlp] {
        train_on(model, dev, &train, &val)?;
    }
    Ok(())
}

#[test]
fn containers_expose_child_buffers() -> Result<()> {
    let seq = Sequential::new()
        .push(Linear::new(8, 8)?)
        .push(BatchNorm::new(8)?);
    assert_eq!(seq.parameters().len(), 4);
    assert_eq!(seq.buffers().len(), 2);

    assert_eq!(Residual::new(BatchNorm::new(8)?).buffers().len(), 2);

    let cat = ConcatBranches::new(1)
        .branch(BatchNorm::new(8)?)
        .branch(BatchNorm::new(8)?);
    assert_eq!(cat.buffers().len(), 4);

    let nested = Sequential::new().push(Sequential::new().push(BatchNorm::new(8)?));
    assert_eq!(nested.parameters().len(), 2);
    assert_eq!(nested.buffers().len(), 2);
    Ok(())
}

#[test]
fn tied_logits_are_not_counted_correct() -> Result<()> {
    let (_, val) = motif_data(false);
    let constant = Lambda::new("constant", |x| {
        let b = x.shape()[0];
        x.reshape(&[b, -1])?.narrow(1, 0, 2)?.mul_scalar(0.0)
    });
    // All logits tie, so every row predicts class 0.
    let class0 = val.y.iter().filter(|&&y| y == 0).count() as f64 / val.n as f64;
    let (_, acc) = evaluate(&constant, &val, 32)?;
    assert!(
        (acc - class0).abs() < 1e-9,
        "accuracy {acc}, class-0 share {class0}"
    );
    assert!((report(&constant, &val, 2, 32)?.accuracy - class0).abs() < 1e-9);
    Ok(())
}

#[test]
fn fit_with_reports_every_epoch() -> Result<()> {
    let (train, val) = motif_data(true);
    let model = MotifCnn::new(&MotifCnnConfig {
        in_channels: 4,
        filters: 8,
        kernel_sizes: vec![6],
        dropout: 0.0,
        classes: 2,
    })?;
    let cfg = TrainConfig {
        epochs: 3,
        lr_decay: Some(0.5),
        verbose: false,
        ..Default::default()
    };
    let mut seen = Vec::new();
    let history = fit_with(
        &model,
        &mut adam(&model, 1e-3),
        &train,
        Some(&val),
        &cfg,
        |s| seen.push((s.epoch, s.lr, s.secs)),
    )?;
    assert_eq!(seen.len(), history.len());
    for ((epoch, lr, secs), h) in seen.iter().zip(&history) {
        assert_eq!(*epoch, h.epoch);
        assert_eq!(*lr, h.lr);
        assert!(*secs > 0.0);
    }
    let lrs: Vec<f64> = seen.iter().map(|s| s.1).collect();
    for (got, want) in lrs.iter().zip([1e-3, 5e-4, 2.5e-4]) {
        assert!((got - want).abs() < 1e-12, "lr per epoch {lrs:?}");
    }
    Ok(())
}

#[test]
fn motif_cnn_filters_recover_the_planted_motif() -> Result<()> {
    use flodl_bio::motif::pfms_from_activations;
    manual_seed(1);
    let (seqs, labels) = synthetic_motif(2000, 50, b"TATAAA", 7);
    let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
    let (train, val) = Dataset::from_sequences(&refs, &labels, &Alphabet::DNA, 50, true)
        .unwrap()
        .split(0.2, 1);
    let model = MotifCnn::new(&MotifCnnConfig {
        in_channels: 4,
        filters: 8,
        kernel_sizes: vec![8],
        dropout: 0.0,
        classes: 2,
    })?;
    let cfg = TrainConfig {
        epochs: 10,
        verbose: false,
        ..Default::default()
    };
    fit(&model, &mut adam(&model, 1e-2), &train, None, &cfg)?;

    let x = Variable::new(
        Tensor::from_f32(&val.x, &[val.n as i64, 4, 50], Device::CPU)?,
        false,
    );
    let acts = model.scan(0, &x).unwrap()?.data().to_f32_vec()?;
    let pfms = pfms_from_activations(&val.x, val.n, 4, 50, &acts, 8, 8, 0.5);
    let consensus: Vec<String> = pfms.iter().map(|p| p.consensus(b"ACGT")).collect();
    assert!(
        consensus.iter().any(|c| c.contains("TATAAA")),
        "no filter learned TATAAA: {consensus:?}"
    );
    Ok(())
}

#[test]
fn models_train_on_cpu() -> Result<()> {
    train_every_model(Device::CPU)
}

#[test]
#[ignore = "needs a GPU build: cargo test --features cuda --test models -- --ignored"]
fn models_train_on_gpu() -> Result<()> {
    assert!(gpu_available(), "no GPU visible to libtorch");
    train_every_model(Device::CUDA(0))
}
