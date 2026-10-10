//! Reproduces the evaluation of the bio-flodl application note. See README.md.
//!
//! Every run executes in its own process (the sweep re-invokes this binary with `run-one`),
//! so process memory and GPU peak statistics belong to that run alone.

mod run;

use bio_flodl::alphabet::Alphabet;
use bio_flodl::data::{synthetic_motif, Dataset};
use bio_flodl::prelude::*;
use run::{device_label, Res, RunSpec};
use serde_json::Value;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const USAGE: &str = "\
usage: bio-flodl-experiments motif [options]

  motif    planted-motif task: random DNA, positives carry TATAAA

options:
  --models a,b,...  motif, dilated, lstm, gru, transformer, transformer-stem, mlp (default: all)
  --device d,...    cpu, cuda (default: cpu)
  --seeds N         seeds 1 to N (default: 5)
  --epochs N        (default: 20)
  --batch N         (default: 64)
  --n N             number of sequences (default: 2000)
  --len N           sequence length (default: 100)
  --out DIR         (default: runs)";

const MODELS: [&str; 7] = [
    "motif",
    "dilated",
    "lstm",
    "gru",
    "transformer",
    "transformer-stem",
    "mlp",
];

/// Columns of `runs.csv`, read from each run's `run.json`.
const RESULT_FIELDS: [&str; 9] = [
    "accuracy",
    "macro_f1",
    "mcc",
    "auroc",
    "auprc",
    "secs_per_epoch",
    "samples_per_sec",
    "max_rss_mb",
    "peak_vram_mb",
];

struct Args {
    task: String,
    models: Vec<String>,
    devices: Vec<String>,
    seeds: u64,
    epochs: usize,
    batch: usize,
    n: usize,
    len: usize,
    out: PathBuf,
}

fn parse_args() -> Res<Args> {
    let mut it = std::env::args().skip(1);
    let task = it.next().ok_or(USAGE)?;
    if task == "-h" || task == "--help" {
        println!("{USAGE}");
        std::process::exit(0);
    }
    let mut a = Args {
        task,
        models: MODELS.iter().map(|m| m.to_string()).collect(),
        devices: vec!["cpu".into()],
        seeds: 5,
        epochs: 20,
        batch: 64,
        n: 2000,
        len: 100,
        out: PathBuf::from("runs"),
    };
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value\n\n{USAGE}"))?;
        let list = || value.split(',').map(str::to_string).collect::<Vec<_>>();
        match flag.as_str() {
            "--models" | "--model" => a.models = list(),
            "--device" => a.devices = list(),
            "--seeds" | "--seed" => a.seeds = value.parse()?,
            "--epochs" => a.epochs = value.parse()?,
            "--batch" => a.batch = value.parse()?,
            "--n" => a.n = value.parse()?,
            "--len" => a.len = value.parse()?,
            "--out" => a.out = PathBuf::from(value),
            _ => return Err(format!("unknown option {flag}\n\n{USAGE}").into()),
        }
    }
    for m in &a.models {
        if !MODELS.contains(&m.as_str()) {
            return Err(format!("unknown model {m} (expected {})", MODELS.join(", ")).into());
        }
    }
    for d in &a.devices {
        parse_device(d)?;
    }
    Ok(a)
}

fn parse_device(name: &str) -> Res<Device> {
    match name {
        "cpu" => Ok(Device::CPU),
        "cuda" if gpu_available() => Ok(Device::CUDA(0)),
        "cuda" => Err("no GPU visible to libtorch (build with --features cuda)".into()),
        other => Err(format!("unknown device {other} (expected cpu or cuda)").into()),
    }
}

/// Builds model `name` for sequences of `len` bases: the model, whether it takes the
/// channels-first layout, and its learning rate (those of `examples/tf_binding.rs`).
fn build(name: &str, len: usize) -> Res<(Box<dyn Module>, bool, f64)> {
    let transformer = |alphabet| TransformerConfig {
        alphabet,
        d_model: 64,
        heads: 4,
        layers: 2,
        d_ff: 128,
        max_len: len,
        dropout: 0.1,
        classes: 2,
    };
    let built: (Box<dyn Module>, bool, f64) = match name {
        "motif" => (
            Box::new(MotifCnn::new(&MotifCnnConfig {
                in_channels: 4,
                filters: 64,
                kernel_sizes: vec![8, 12, 16],
                dropout: 0.3,
                classes: 2,
            })?),
            true,
            1e-3,
        ),
        "dilated" => (
            Box::new(DilatedResCnn::new(&DilatedCnnConfig {
                in_channels: 4,
                channels: 64,
                kernel: 3,
                blocks: 6,
                dropout: 0.3,
                classes: 2,
            })?),
            true,
            1e-3,
        ),
        "lstm" => (
            Box::new(BiRnnClassifier::new(RnnKind::Lstm, 4, 64, 1, 0.3, 2)?),
            false,
            2e-3,
        ),
        "gru" => (
            Box::new(BiRnnClassifier::new(RnnKind::Gru, 4, 64, 1, 0.3, 2)?),
            false,
            2e-3,
        ),
        "transformer" => (
            Box::new(TransformerClassifier::new(&transformer(4))?),
            false,
            5e-4,
        ),
        // The same transformer behind a Conv1d stem, which gives it local k-mer features.
        "transformer-stem" => (
            Box::new(
                Sequential::new()
                    .push(Lambda::transpose(1, 2))
                    .push(Conv1d::new(4, 64, 9)?)
                    .push(Lambda::relu())
                    .push(Lambda::transpose(1, 2))
                    .push(TransformerClassifier::new(&transformer(64))?),
            ),
            false,
            5e-4,
        ),
        "mlp" => (
            Box::new(FlatMlp::new(4 * len as i64, &[256, 256], 2, 0.3)?),
            false,
            1e-3,
        ),
        other => return Err(format!("unknown model {other}").into()),
    };
    Ok(built)
}

fn task_name(a: &Args) -> String {
    format!("motif-n{}-len{}", a.n, a.len)
}

fn run_dir(a: &Args, model: &str, device: &str, seed: u64) -> PathBuf {
    a.out
        .join(task_name(a))
        .join(model)
        .join(device)
        .join(format!("seed-{seed}"))
}

/// Trains one (model, device, seed) cell in this process.
fn run_one(a: &Args) -> Res<()> {
    let (model_name, device, seed) = (&a.models[0], parse_device(&a.devices[0])?, a.seeds);
    let (seqs, labels) = synthetic_motif(a.n, a.len, b"TATAAA", 42);
    let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
    manual_seed(seed);
    let (model, channels_first, lr) = build(model_name, a.len)?;
    let (train, val) =
        Dataset::from_sequences(&refs, &labels, &Alphabet::DNA, a.len, channels_first)?
            .split(0.2, 1);
    let task = task_name(a);
    let spec = RunSpec {
        task: &task,
        model: model_name,
        device,
        seed,
        lr,
        epochs: a.epochs,
        batch: a.batch,
        train: &train,
        val: &val,
        dir: run_dir(a, model_name, &device_label(device), seed),
    };
    let r = run::run(model.as_ref(), &spec)?;
    println!(
        "{model_name} {} seed {seed}: accuracy {:.4}  MCC {:.4}  AUROC {:.4}  AUPRC {:.4}  {:.2} s/epoch",
        r.device, r.accuracy, r.mcc, r.auroc, r.auprc, r.secs_per_epoch
    );
    Ok(())
}

/// Runs every (model, device, seed) cell in a child process, then writes the summary.
fn sweep(a: &Args) -> Res<()> {
    let exe = std::env::current_exe()?;
    let total = a.models.len() * a.devices.len() * a.seeds as usize;
    let mut cell = 0;
    let mut rows = Vec::new();
    for model in &a.models {
        for device in &a.devices {
            for seed in 1..=a.seeds {
                cell += 1;
                println!("[{cell}/{total}] {model} {device} seed {seed}");
                let status = Command::new(&exe)
                    .args(["run-one", "--model", model, "--device", device])
                    .args(["--seed", &seed.to_string()])
                    .args(["--epochs", &a.epochs.to_string()])
                    .args(["--batch", &a.batch.to_string()])
                    .args(["--n", &a.n.to_string(), "--len", &a.len.to_string()])
                    .arg("--out")
                    .arg(&a.out)
                    .status()?;
                if !status.success() {
                    return Err(format!("{model} {device} seed {seed} failed ({status})").into());
                }
                let label = device_label(parse_device(device)?);
                let json = fs::read_to_string(run_dir(a, model, &label, seed).join("run.json"))?;
                rows.push(serde_json::from_str::<Value>(&json)?);
            }
        }
    }
    let root = a.out.join(task_name(a));
    write_summary(&root, &rows, a.seeds)?;
    println!("summary: {}", root.join("table.md").display());
    Ok(())
}

fn field(row: &Value, name: &str) -> Option<f64> {
    row["result"][name].as_f64()
}

/// `mean ± sample sd`, or the mean alone for a single value.
fn mean_sd(values: &[f64], digits: usize) -> String {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return format!("{mean:.digits$}");
    }
    let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    format!("{mean:.digits$} ± {sd:.digits$}")
}

/// `runs.csv` holds one row per run; `table.md` aggregates them per model and device.
fn write_summary(root: &Path, rows: &[Value], seeds: u64) -> Res<()> {
    let mut csv = format!("model,device,seed,params,{}\n", RESULT_FIELDS.join(","));
    for row in rows {
        write!(
            csv,
            "{},{},{},{}",
            row["model"], row["device"], row["seed"], row["params"]
        )?;
        for f in RESULT_FIELDS {
            match field(row, f) {
                Some(v) => write!(csv, ",{v}")?,
                None => csv.push(','),
            }
        }
        csv.push('\n');
    }
    fs::write(root.join("runs.csv"), csv.replace('"', ""))?;

    let mut md = format!(
        "{seeds} seed(s) per cell, mean ± sample sd. Memory: peak RSS sampled at epoch ends, \
         peak GPU memory allocated to tensors.\n\n\
         | model | device | params | accuracy | MCC | AUROC | AUPRC | s/epoch | RSS MB | GPU MB |\n\
         |---|---|---|---|---|---|---|---|---|---|\n"
    );
    let mut cells: Vec<(String, String)> = Vec::new();
    for row in rows {
        let key = (row["model"].to_string(), row["device"].to_string());
        if !cells.contains(&key) {
            cells.push(key);
        }
    }
    for (model, device) in &cells {
        let group: Vec<&Value> = rows
            .iter()
            .filter(|r| &r["model"].to_string() == model && &r["device"].to_string() == device)
            .collect();
        let col = |name: &str, digits| {
            let values: Vec<f64> = group.iter().filter_map(|r| field(r, name)).collect();
            if values.is_empty() {
                "-".to_string()
            } else {
                mean_sd(&values, digits)
            }
        };
        writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            model.trim_matches('"'),
            device.trim_matches('"'),
            group[0]["params"],
            col("accuracy", 4),
            col("mcc", 4),
            col("auroc", 4),
            col("auprc", 4),
            col("secs_per_epoch", 3),
            col("max_rss_mb", 0),
            col("peak_vram_mb", 0),
        )?;
    }
    fs::write(root.join("table.md"), md)?;
    Ok(())
}

fn main() -> Res<()> {
    let a = parse_args()?;
    match a.task.as_str() {
        "motif" => sweep(&a),
        "run-one" => run_one(&a),
        other => Err(format!("unknown task {other}\n\n{USAGE}").into()),
    }
}
