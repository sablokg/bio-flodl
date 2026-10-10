//! Reproduces the evaluation of the flodl-bio application note. See README.md.
//!
//! Every run executes in its own process (the sweep re-invokes this binary with `run-one`),
//! so process memory and GPU peak statistics belong to that run alone.

mod run;

use flodl_bio::alphabet::Alphabet;
use flodl_bio::data::{synthetic_motif, Dataset};
use flodl_bio::prelude::*;
use run::{device_label, Res, RunSpec};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const USAGE: &str = "\
usage: flodl-bio-experiments <motif|gue> [options]

  motif    planted-motif task: random DNA, positives carry TATAAA
  gue      GUE human transcription-factor binding (DNABERT-2 benchmark, 5 ENCODE ChIP-seq
           datasets): train on train, follow dev, report on test

options:
  --models a,b,...    motif, dilated, lstm, gru, transformer, transformer-stem, mlp (default: all)
  --device d,...      cpu, cuda (default: cpu)
  --seeds N           seeds 1 to N (default: 5)
  --epochs N          (default: 20)
  --batch N           (default: 64)
  --out DIR           (default: runs)
motif:
  --n N               number of sequences (default: 2000)
  --len N             sequence length (default: 100)
gue:
  --data DIR          the extracted GUE directory (the one holding tf/)
  --datasets k,...    TF datasets 0 to 4 (default: all)";

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
    /// `motif`, `gue`, or `run-one` (one cell, in a child process).
    command: String,
    /// `motif` or `gue`; for `run-one` it comes from `--task`.
    task: String,
    models: Vec<String>,
    devices: Vec<String>,
    seeds: u64,
    epochs: usize,
    batch: usize,
    n: usize,
    len: usize,
    data: Option<PathBuf>,
    datasets: Vec<usize>,
    out: PathBuf,
}

fn parse_args() -> Res<Args> {
    let mut it = std::env::args().skip(1);
    let command = it.next().ok_or(USAGE)?;
    if command == "-h" || command == "--help" {
        println!("{USAGE}");
        std::process::exit(0);
    }
    let mut a = Args {
        task: command.clone(),
        command,
        models: MODELS.iter().map(|m| m.to_string()).collect(),
        devices: vec!["cpu".into()],
        seeds: 5,
        epochs: 20,
        batch: 64,
        n: 2000,
        len: 100,
        data: None,
        datasets: (0..5).collect(),
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
            "--task" => a.task = value,
            "--data" => a.data = Some(PathBuf::from(value)),
            "--datasets" | "--dataset" => {
                a.datasets = list()
                    .iter()
                    .map(|k| k.parse())
                    .collect::<std::result::Result<_, _>>()?
            }
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
    if a.task == "gue" && a.data.is_none() {
        return Err(format!("gue needs --data <GUE dir>\n\n{USAGE}").into());
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

/// Whether model `name` takes `[batch, alphabet, length]` (the convolutional models) rather
/// than `[batch, length, alphabet]`.
fn channels_first(name: &str) -> bool {
    matches!(name, "motif" | "dilated")
}

/// A built model. MotifCnn stays concrete so its filters can be exported after training.
enum Built {
    Motif(MotifCnn),
    Other(Box<dyn Module>),
}

impl Built {
    fn module(&self) -> &dyn Module {
        match self {
            Built::Motif(m) => m,
            Built::Other(m) => m.as_ref(),
        }
    }
}

/// Builds model `name` for sequences of `len` bases, with its learning rate (those of
/// `examples/tf_binding.rs`).
fn build(name: &str, len: usize) -> Res<(Built, f64)> {
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
    if name == "motif" {
        let motif = MotifCnn::new(&MotifCnnConfig {
            in_channels: 4,
            filters: 64,
            kernel_sizes: vec![8, 12, 16],
            dropout: 0.3,
            classes: 2,
        })?;
        return Ok((Built::Motif(motif), 1e-3));
    }
    let (model, lr): (Box<dyn Module>, f64) = match name {
        "dilated" => (
            Box::new(DilatedResCnn::new(&DilatedCnnConfig {
                in_channels: 4,
                channels: 64,
                kernel: 3,
                blocks: 6,
                dropout: 0.3,
                classes: 2,
            })?),
            1e-3,
        ),
        "lstm" => (
            Box::new(BiRnnClassifier::new(RnnKind::Lstm, 4, 64, 1, 0.3, 2)?),
            2e-3,
        ),
        "gru" => (
            Box::new(BiRnnClassifier::new(RnnKind::Gru, 4, 64, 1, 0.3, 2)?),
            2e-3,
        ),
        "transformer" => (Box::new(TransformerClassifier::new(&transformer(4))?), 5e-4),
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
            5e-4,
        ),
        "mlp" => (
            Box::new(FlatMlp::new(4 * len as i64, &[256, 256], 2, 0.3)?),
            1e-3,
        ),
        other => return Err(format!("unknown model {other}").into()),
    };
    Ok((Built::Other(model), lr))
}

/// One dataset of a task: its name (the output directory) and, for GUE, its index.
struct Cell {
    name: String,
    dataset: Option<usize>,
}

fn cells(a: &Args) -> Vec<Cell> {
    match a.task.as_str() {
        "gue" => a
            .datasets
            .iter()
            .map(|&k| Cell {
                name: format!("gue-tf-{k}"),
                dataset: Some(k),
            })
            .collect(),
        _ => vec![Cell {
            name: format!("motif-n{}-len{}", a.n, a.len),
            dataset: None,
        }],
    }
}

fn run_dir(a: &Args, cell: &str, model: &str, device: &str, seed: u64) -> PathBuf {
    a.out
        .join(cell)
        .join(model)
        .join(device)
        .join(format!("seed-{seed}"))
}

/// Train, validation and (GUE only) test sets in the layout a model expects, and the
/// sequence length.
struct Splits {
    train: Dataset,
    val: Dataset,
    test: Option<Dataset>,
    len: usize,
}

/// `sequence,label` rows after a header line.
fn read_csv(path: &Path) -> Res<(Vec<Vec<u8>>, Vec<i64>)> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut seqs, mut labels) = (Vec::new(), Vec::new());
    for (i, line) in text.lines().enumerate().skip(1) {
        let (seq, label) = line
            .split_once(',')
            .ok_or_else(|| format!("{}:{}: expected sequence,label", path.display(), i + 1))?;
        seqs.push(seq.as_bytes().to_vec());
        labels.push(label.trim().parse()?);
    }
    Ok((seqs, labels))
}

fn load(a: &Args, cell: &Cell, channels_first: bool) -> Res<Splits> {
    let encode = |seqs: &[Vec<u8>], labels: &[i64], len| -> Res<Dataset> {
        let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
        Ok(Dataset::from_sequences(
            &refs,
            labels,
            &Alphabet::DNA,
            len,
            channels_first,
        )?)
    };
    match cell.dataset {
        None => {
            let (seqs, labels) = synthetic_motif(a.n, a.len, b"TATAAA", 42);
            let (train, val) = encode(&seqs, &labels, a.len)?.split(0.2, 1);
            Ok(Splits {
                train,
                val,
                test: None,
                len: a.len,
            })
        }
        Some(k) => {
            let dir = a
                .data
                .as_ref()
                .ok_or("gue needs --data")?
                .join(format!("tf/{k}"));
            let parts =
                ["train", "dev", "test"].map(|split| read_csv(&dir.join(format!("{split}.csv"))));
            let [train, dev, test] = parts;
            let (train, dev, test) = (train?, dev?, test?);
            let len = [&train, &dev, &test]
                .iter()
                .flat_map(|(seqs, _)| seqs.iter().map(|s| s.len()))
                .max()
                .unwrap_or(0);
            Ok(Splits {
                train: encode(&train.0, &train.1, len)?,
                val: encode(&dev.0, &dev.1, len)?,
                test: Some(encode(&test.0, &test.1, len)?),
                len,
            })
        }
    }
}

/// Trains one (dataset, model, device, seed) cell in this process.
fn run_one(a: &Args) -> Res<()> {
    let (model_name, device, seed) = (&a.models[0], parse_device(&a.devices[0])?, a.seeds);
    let cell = cells(a).remove(0);
    let data = load(a, &cell, channels_first(model_name))?;
    manual_seed(seed);
    let (model, lr) = build(model_name, data.len)?;
    let spec = RunSpec {
        task: &cell.name,
        model: model_name,
        device,
        seed,
        lr,
        epochs: a.epochs,
        batch: a.batch,
        train: &data.train,
        val: &data.val,
        test: data.test.as_ref(),
        dir: run_dir(a, &cell.name, model_name, &device_label(device), seed),
    };
    let r = run::run(model.module(), &spec)?;
    if let Built::Motif(motif) = &model {
        run::write_filters(motif, data.test.as_ref().unwrap_or(&data.val), &spec.dir)?;
    }
    println!(
        "{} {model_name} {} seed {seed}: accuracy {:.4}  MCC {:.4}  AUROC {:.4}  AUPRC {:.4}  {:.2} s/epoch",
        cell.name, r.device, r.accuracy, r.mcc, r.auroc, r.auprc, r.secs_per_epoch
    );
    Ok(())
}

/// Runs every (dataset, model, device, seed) cell in a child process, then writes the
/// summaries.
fn sweep(a: &Args) -> Res<()> {
    let exe = std::env::current_exe()?;
    let cells = cells(a);
    let total = cells.len() * a.models.len() * a.devices.len() * a.seeds as usize;
    let mut done = 0;
    let mut all_rows = Vec::new();
    for cell in &cells {
        let mut rows = Vec::new();
        for model in &a.models {
            for device in &a.devices {
                for seed in 1..=a.seeds {
                    done += 1;
                    println!(
                        "[{done}/{total}] {} {model} {device} seed {seed}",
                        cell.name
                    );
                    let mut cmd = Command::new(&exe);
                    cmd.args(["run-one", "--task", &a.task, "--model", model])
                        .args(["--device", device, "--seed", &seed.to_string()])
                        .args(["--epochs", &a.epochs.to_string()])
                        .args(["--batch", &a.batch.to_string()])
                        .args(["--n", &a.n.to_string(), "--len", &a.len.to_string()])
                        .arg("--out")
                        .arg(&a.out);
                    if let (Some(k), Some(data)) = (cell.dataset, &a.data) {
                        cmd.args(["--dataset", &k.to_string()])
                            .arg("--data")
                            .arg(data);
                    }
                    let status = cmd.status()?;
                    if !status.success() {
                        return Err(format!(
                            "{} {model} {device} seed {seed} failed ({status})",
                            cell.name
                        )
                        .into());
                    }
                    let label = device_label(parse_device(device)?);
                    let dir = run_dir(a, &cell.name, model, &label, seed);
                    let json = fs::read_to_string(dir.join("run.json"))?;
                    rows.push(serde_json::from_str::<Value>(&json)?);
                }
            }
        }
        write_summary(&a.out.join(&cell.name), &rows, a.seeds)?;
        all_rows.extend(rows);
    }
    if cells.len() > 1 {
        let path = a.out.join(format!("{}-overview.md", a.task));
        write_overview(&path, &all_rows, &cells, a.seeds)?;
        println!("overview: {}", path.display());
    }
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

/// MCC and AUROC per dataset (columns) for each model and device (rows): the paper's table.
fn write_overview(path: &Path, rows: &[Value], cells: &[Cell], seeds: u64) -> Res<()> {
    let names: Vec<&str> = cells.iter().map(|c| c.name.as_str()).collect();
    let mut groups: BTreeMap<(String, String), Vec<&Value>> = BTreeMap::new();
    let mut order = Vec::new();
    for row in rows {
        let key = (
            row["model"].as_str().unwrap_or_default().to_string(),
            row["device"].as_str().unwrap_or_default().to_string(),
        );
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(row);
    }
    let mut md = String::new();
    for (metric, label) in [("mcc", "MCC"), ("auroc", "AUROC")] {
        writeln!(
            md,
            "## {label} on the test split, {seeds} seed(s), mean ± sample sd\n"
        )?;
        writeln!(md, "| model | device | {} |", names.join(" | "))?;
        writeln!(md, "|---|---|{}", "---|".repeat(names.len()))?;
        for key in &order {
            let group = &groups[key];
            let cols: Vec<String> = names
                .iter()
                .map(|name| {
                    let values: Vec<f64> = group
                        .iter()
                        .filter(|r| r["task"].as_str() == Some(*name))
                        .filter_map(|r| field(r, metric))
                        .collect();
                    if values.is_empty() {
                        "-".to_string()
                    } else {
                        mean_sd(&values, 3)
                    }
                })
                .collect();
            writeln!(md, "| {} | {} | {} |", key.0, key.1, cols.join(" | "))?;
        }
        md.push('\n');
    }
    fs::write(path, md)?;
    Ok(())
}

fn main() {
    let result = parse_args().and_then(|a| match a.command.as_str() {
        "motif" | "gue" => sweep(&a),
        "run-one" => run_one(&a),
        other => Err(format!("unknown task {other}\n\n{USAGE}").into()),
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
