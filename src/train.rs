//! Training and evaluation loops for classification (feature `flodl`).
//!
//! Works with any `flodl::Module` and any `flodl::Optimizer` (Adam, AdamW, SGD, ...).
//! Targets are class indices; the loss is `cross_entropy_loss` on raw logits.
//! Batches are built on the device of the model's parameters, so calling
//! `model.move_to_device(...)` before training is all a GPU run needs.

use crate::data::{
    auroc, average_precision, batch_indices, confusion_matrix, macro_f1, mcc_binary, Dataset, Rng,
};
use flodl::{
    clip_grad_norm, cross_entropy_loss, Adam, Device, Module, Optimizer, Tensor, Variable,
};
use std::time::Instant;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    /// Multiply the learning rate by this each epoch (e.g. 0.95). `None` keeps it constant.
    pub lr_decay: Option<f64>,
    /// Global gradient-norm clipping.
    pub clip_norm: Option<f64>,
    /// Stop after this many epochs without validation-loss improvement.
    pub patience: Option<usize>,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            epochs: 20,
            batch_size: 32,
            lr_decay: None,
            clip_norm: Some(1.0),
            patience: None,
            seed: 0,
            verbose: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpochStats {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_acc: f64,
    pub val_loss: Option<f64>,
    pub val_acc: Option<f64>,
    /// Learning rate used during the epoch.
    pub lr: f64,
    /// Wall time of the epoch in seconds, validation included.
    pub secs: f64,
}

/// Convenience: Adam over all of the model's parameters.
pub fn adam<M: Module + ?Sized>(model: &M, lr: f64) -> Adam {
    Adam::new(&model.parameters(), lr)
}

/// Device of the model's first parameter, CPU for a parameter-free model.
fn model_device<M: Module + ?Sized>(model: &M) -> Device {
    model
        .parameters()
        .first()
        .map_or(Device::CPU, |p| p.variable.device())
}

/// Inputs and class-index targets for the samples `idx`, created on `device`.
fn batch_inputs(
    data: &Dataset,
    idx: &[usize],
    device: Device,
) -> flodl::Result<(Variable, Variable)> {
    let (x, y) = data.batch(idx);
    let mut shape = vec![idx.len() as i64];
    shape.extend(data.sample_shape.iter().map(|&d| d as i64));
    let x = Variable::new(Tensor::from_f32(&x, &shape, device)?, false);
    let t = Variable::new(Tensor::from_i64(&y, &[y.len() as i64], device)?, false);
    Ok((x, t))
}

/// Number of rows whose highest logit is the true class. On ties the lowest class index
/// wins, as in [`report`].
fn count_correct(logits: &Variable, targets: &Variable) -> flodl::Result<f64> {
    logits
        .data()
        .argmax(1, false)?
        .eq_tensor(&targets.data())?
        .sum()?
        .item()
}

/// Mean loss and accuracy over `data`. Switches the model to eval mode.
pub fn evaluate<M: Module + ?Sized>(
    model: &M,
    data: &Dataset,
    batch_size: usize,
) -> flodl::Result<(f64, f64)> {
    model.eval();
    let device = model_device(model);
    let (mut loss_sum, mut correct) = (0.0, 0.0);
    for idx in batch_indices(data.n, batch_size, None) {
        let (x, t) = batch_inputs(data, &idx, device)?;
        let logits = model.forward(&x)?;
        loss_sum += cross_entropy_loss(&logits, &t)?.item()? * idx.len() as f64;
        correct += count_correct(&logits, &t)?;
    }
    let n = data.n.max(1) as f64;
    Ok((loss_sum / n, correct / n))
}

/// Class probabilities `[batch, classes]` for an input batch (eval mode, softmax over classes).
pub fn predict_proba<M: Module + ?Sized>(model: &M, x: &Variable) -> flodl::Result<Variable> {
    model.eval();
    model.forward(x)?.softmax(1)
}

/// Train `model`, optionally validating each epoch. Returns per-epoch statistics.
pub fn fit<M: Module + ?Sized>(
    model: &M,
    opt: &mut dyn Optimizer,
    train: &Dataset,
    val: Option<&Dataset>,
    cfg: &TrainConfig,
) -> flodl::Result<Vec<EpochStats>> {
    fit_with(model, opt, train, val, cfg, |_| {})
}

/// [`fit`], calling `on_epoch` with each epoch's statistics as soon as the epoch ends,
/// for example to feed a `flodl::Monitor` or write curves while training runs.
pub fn fit_with<M: Module + ?Sized>(
    model: &M,
    opt: &mut dyn Optimizer,
    train: &Dataset,
    val: Option<&Dataset>,
    cfg: &TrainConfig,
    mut on_epoch: impl FnMut(&EpochStats),
) -> flodl::Result<Vec<EpochStats>> {
    let params = model.parameters();
    let device = model_device(model);
    let mut rng = Rng::new(cfg.seed);
    let mut history = Vec::new();
    let (mut best_val, mut since_best) = (f64::INFINITY, 0usize);

    for epoch in 1..=cfg.epochs {
        let started = Instant::now();
        let lr = opt.lr();
        model.train();
        let (mut loss_sum, mut correct) = (0.0, 0.0);
        for idx in batch_indices(train.n, cfg.batch_size, Some(&mut rng)) {
            let (x, t) = batch_inputs(train, &idx, device)?;
            let logits = model.forward(&x)?;
            let loss = cross_entropy_loss(&logits, &t)?;
            opt.zero_grad();
            loss.backward()?;
            if let Some(max_norm) = cfg.clip_norm {
                clip_grad_norm(&params, max_norm)?;
            }
            opt.step()?;
            loss_sum += loss.item()? * idx.len() as f64;
            correct += count_correct(&logits, &t)?;
        }
        let n = train.n.max(1) as f64;
        let (val_loss, val_acc) = match val {
            Some(v) => {
                let (l, a) = evaluate(model, v, cfg.batch_size)?;
                (Some(l), Some(a))
            }
            None => (None, None),
        };
        let stats = EpochStats {
            epoch,
            train_loss: loss_sum / n,
            train_acc: correct / n,
            val_loss,
            val_acc,
            lr,
            secs: started.elapsed().as_secs_f64(),
        };
        if cfg.verbose {
            println!(
                "epoch {:>3}  train loss {:.4} acc {:.3}  val loss {}  val acc {}",
                epoch,
                stats.train_loss,
                stats.train_acc,
                val_loss.map_or("-".into(), |v| format!("{v:.4}")),
                val_acc.map_or("-".into(), |v| format!("{v:.3}"))
            );
        }
        on_epoch(&stats);
        history.push(stats);
        if let Some(f) = cfg.lr_decay {
            opt.scale_lr(f);
        }
        if let (Some(vl), Some(p)) = (val_loss, cfg.patience) {
            if vl < best_val - 1e-6 {
                best_val = vl;
                since_best = 0;
            } else {
                since_best += 1;
                if since_best >= p {
                    break;
                }
            }
        }
    }
    Ok(history)
}

/// Softmax class probabilities for the whole dataset, flattened row-major `[n * classes]`.
pub fn predict_probs<M: Module + ?Sized>(
    model: &M,
    data: &Dataset,
    batch_size: usize,
) -> flodl::Result<Vec<f32>> {
    model.eval();
    let device = model_device(model);
    let mut out = Vec::new();
    for idx in batch_indices(data.n, batch_size, None) {
        let (x, _) = batch_inputs(data, &idx, device)?;
        out.extend(model.forward(&x)?.softmax(1)?.data().to_f32_vec()?);
    }
    Ok(out)
}

/// Held-out evaluation summary. Binary-only fields are `None` when `classes != 2`
/// or when only one class is present.
#[derive(Debug, Clone)]
pub struct Report {
    pub accuracy: f64,
    pub macro_f1: f64,
    pub mcc: Option<f64>,
    pub auroc: Option<f64>,
    pub auprc: Option<f64>,
}

/// Accuracy, macro-F1 and (for binary tasks) MCC, AUROC and AUPRC, computed on the host
/// from softmax probabilities.
pub fn report<M: Module + ?Sized>(
    model: &M,
    data: &Dataset,
    classes: usize,
    batch_size: usize,
) -> flodl::Result<Report> {
    let probs = predict_probs(model, data, batch_size)?;
    let pred: Vec<usize> = probs
        .chunks(classes)
        .map(|r| {
            r.iter()
                .enumerate()
                .fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                .0
        })
        .collect();
    let cm = confusion_matrix(&pred, &data.y, classes);
    let acc = crate::data::accuracy(&pred, &data.y);
    let (mut mcc, mut roc, mut ap) = (None, None, None);
    if classes == 2 {
        let scores: Vec<f32> = probs.chunks(2).map(|r| r[1]).collect();
        mcc = Some(mcc_binary(&cm));
        roc = auroc(&scores, &data.y);
        ap = average_precision(&scores, &data.y);
    }
    Ok(Report {
        accuracy: acc,
        macro_f1: macro_f1(&cm, classes),
        mcc,
        auroc: roc,
        auprc: ap,
    })
}
