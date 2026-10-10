//! One training run: seed, train with a flodl Monitor, then write every artifact of the run.

use bio_flodl::data::{pr_curve, roc_curve, Dataset};
use bio_flodl::prelude::*;
use serde_json::json;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub type Res<T> = std::result::Result<T, Box<dyn Error>>;

/// What a run trains and where its artifacts go.
pub struct RunSpec<'a> {
    pub task: &'a str,
    pub model: &'a str,
    pub device: Device,
    pub seed: u64,
    pub lr: f64,
    pub epochs: usize,
    pub batch: usize,
    pub train: &'a Dataset,
    pub val: &'a Dataset,
    pub dir: PathBuf,
}

/// The numbers a run contributes to the summary table.
pub struct RunResult {
    pub device: String,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub mcc: f64,
    pub auroc: f64,
    pub auprc: f64,
    pub secs_per_epoch: f64,
    pub samples_per_sec: f64,
    pub max_rss_mb: f64,
    pub peak_vram_mb: Option<f64>,
}

pub fn device_label(device: Device) -> String {
    match device {
        Device::CPU => "cpu".into(),
        Device::CUDA(i) => format!("cuda{i}"),
    }
}

fn path_str(dir: &Path, file: &str) -> String {
    dir.join(file).to_string_lossy().into_owned()
}

/// Trains `model` (already built under `manual_seed(spec.seed)`) and writes, under `spec.dir`:
/// `dashboard.html`, `epochs.csv`, `predictions.csv`, `roc.csv`, `pr.csv` and `run.json`.
pub fn run(model: &dyn Module, spec: &RunSpec) -> Res<RunResult> {
    fs::create_dir_all(&spec.dir)?;
    let gpu = spec.device != Device::CPU;
    model.move_to_device(spec.device);
    if gpu {
        gpu_reset_peak_stats();
    }
    let params: usize = model
        .parameters()
        .iter()
        .map(|p| p.variable.shape().iter().product::<i64>() as usize)
        .sum();
    let meta = json!({
        "task": spec.task,
        "model": spec.model,
        "device": device_label(spec.device),
        "seed": spec.seed,
        "lr": spec.lr,
        "epochs": spec.epochs,
        "batch": spec.batch,
        "params": params,
        "train_samples": spec.train.n,
        "val_samples": spec.val.n,
    });

    let mut monitor = Monitor::new(spec.epochs);
    monitor.set_archive_theme("light");
    monitor.save_html(&path_str(&spec.dir, "dashboard.html"));
    monitor.set_metadata(meta.clone());

    let cfg = TrainConfig {
        epochs: spec.epochs,
        batch_size: spec.batch,
        seed: spec.seed,
        verbose: false,
        ..Default::default()
    };
    let mut opt = adam(model, spec.lr);
    let mut max_rss_kb = rss_kb();
    let history = fit_with(model, &mut opt, spec.train, Some(spec.val), &cfg, |s| {
        let metrics = [
            ("train_loss", s.train_loss),
            ("train_acc", s.train_acc),
            ("val_loss", s.val_loss.unwrap_or(f64::NAN)),
            ("val_acc", s.val_acc.unwrap_or(f64::NAN)),
            ("lr", s.lr),
        ];
        monitor.log(s.epoch - 1, Duration::from_secs_f64(s.secs), &metrics);
        max_rss_kb = max_rss_kb.max(rss_kb());
    })?;
    monitor.silent_summary();
    monitor.finish();
    monitor.export_csv(&path_str(&spec.dir, "epochs.csv"))?;
    let peak_vram_mb = gpu
        .then(|| gpu_peak_active_bytes().map(|b| b as f64 / 1048576.0))
        .transpose()?;

    let probs = predict_probs(model, spec.val, spec.batch)?;
    let scores: Vec<f32> = probs.chunks(2).map(|r| r[1]).collect();
    let mut predictions = String::from("index,label,p_positive\n");
    for (i, (label, p)) in spec.val.y.iter().zip(&scores).enumerate() {
        writeln!(predictions, "{i},{label},{p}")?;
    }
    fs::write(spec.dir.join("predictions.csv"), predictions)?;
    write_points(
        &spec.dir.join("roc.csv"),
        "fpr,tpr",
        roc_curve(&scores, &spec.val.y),
    )?;
    write_points(
        &spec.dir.join("pr.csv"),
        "recall,precision",
        pr_curve(&scores, &spec.val.y),
    )?;

    let r = report(model, spec.val, 2, spec.batch)?;
    let secs: f64 = history.iter().map(|h| h.secs).sum();
    let secs_per_epoch = secs / history.len().max(1) as f64;
    let result = RunResult {
        device: device_label(spec.device),
        accuracy: r.accuracy,
        macro_f1: r.macro_f1,
        mcc: r.mcc.unwrap_or(f64::NAN),
        auroc: r.auroc.unwrap_or(f64::NAN),
        auprc: r.auprc.unwrap_or(f64::NAN),
        secs_per_epoch,
        samples_per_sec: spec.train.n as f64 / secs_per_epoch,
        max_rss_mb: max_rss_kb as f64 / 1024.0,
        peak_vram_mb,
    };

    let mut record = meta;
    record["hardware"] = json!(hardware_summary());
    record["bio_flodl_commit"] = json!(git_commit());
    record["result"] = json!({
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "mcc": result.mcc,
        "auroc": result.auroc,
        "auprc": result.auprc,
        "secs_per_epoch": result.secs_per_epoch,
        "samples_per_sec": result.samples_per_sec,
        "max_rss_mb": result.max_rss_mb,
        "peak_vram_mb": result.peak_vram_mb,
    });
    fs::write(
        spec.dir.join("run.json"),
        serde_json::to_string_pretty(&record)?,
    )?;
    Ok(result)
}

fn write_points(path: &Path, header: &str, points: Option<Vec<(f64, f64)>>) -> Res<()> {
    let mut out = format!("{header}\n");
    for (x, y) in points.unwrap_or_default() {
        writeln!(out, "{x},{y}")?;
    }
    fs::write(path, out)?;
    Ok(())
}

/// Commit of the bio-flodl checkout the harness was built from, suffixed `-dirty` when the
/// tree had uncommitted changes, when git can tell.
fn git_commit() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["describe", "--always", "--dirty", "--abbrev=40"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}
