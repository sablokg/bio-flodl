//! Benchmark harness for real data: compare every sequence model on a positive/negative FASTA pair
//! (e.g. ChIP-seq/ENCODE peaks vs. matched background). This is the "TF-binding prediction" benchmark.
//!
//!   cargo run --release --features flodl --example tf_binding -- pos.fa neg.fa \
//!       [--len 200] [--epochs 20] [--batch 64] [--seed 1] [--models motif,dilated,lstm,gru,transformer]
//!
//! Sequences are centre-cropped / N-padded to `--len`. The split is 80/20, seeded, identical for every model.

mod common;
use bio_flodl::alphabet::Alphabet;
use bio_flodl::data::Dataset;
use bio_flodl::io::{fit_length, read_fasta};
use bio_flodl::models::cnn::{DilatedCnnConfig, DilatedResCnn, MotifCnn, MotifCnnConfig};
use bio_flodl::models::rnn::{BiRnnClassifier, RnnKind};
use bio_flodl::models::transformer::{TransformerClassifier, TransformerConfig};
use bio_flodl::train::{adam, fit, report, TrainConfig};
use flodl::Module;
use std::time::Instant;

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.into())
}

fn build(name: &str, len: usize) -> flodl::Result<(Box<dyn Module>, bool, f64)> {
    // returns (model, channels_first, learning rate)
    Ok(match name {
        "motif" => (Box::new(MotifCnn::new(&MotifCnnConfig { in_channels: 4, filters: 64, kernel_sizes: vec![8, 12, 16], dropout: 0.3, classes: 2 })?), true, 1e-3),
        "dilated" => (Box::new(DilatedResCnn::new(&DilatedCnnConfig { in_channels: 4, channels: 64, kernel: 3, blocks: 6, dropout: 0.3, classes: 2 })?), true, 1e-3),
        "lstm" => (Box::new(BiRnnClassifier::new(RnnKind::Lstm, 4, 64, 1, 0.3, 2)?), false, 2e-3),
        "gru" => (Box::new(BiRnnClassifier::new(RnnKind::Gru, 4, 64, 1, 0.3, 2)?), false, 2e-3),
        "transformer" => (Box::new(TransformerClassifier::new(&TransformerConfig { alphabet: 4, d_model: 64, heads: 4, layers: 2, d_ff: 128, max_len: len, dropout: 0.1, classes: 2 })?), false, 5e-4),
        other => panic!("unknown model '{other}' (motif, dilated, lstm, gru, transformer)"),
    })
}

fn main() -> flodl::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let files: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    // positional args are the two FASTA files; flag values are consumed by `arg`
    let positional: Vec<&String> = files.iter().copied().filter(|f| f.ends_with(".fa") || f.ends_with(".fasta") || f.ends_with(".fna")).collect();
    if positional.len() < 2 {
        eprintln!("usage: tf_binding pos.fa neg.fa [--len N] [--epochs N] [--batch N] [--seed N] [--models a,b,c]");
        std::process::exit(2);
    }
    let len: usize = arg(&args, "--len", "200").parse().unwrap();
    let epochs: usize = arg(&args, "--epochs", "20").parse().unwrap();
    let batch: usize = arg(&args, "--batch", "64").parse().unwrap();
    let seed: u64 = arg(&args, "--seed", "1").parse().unwrap();
    let models = arg(&args, "--models", "motif,dilated,lstm,gru,transformer");

    let pos = read_fasta(positional[0]).expect("reading positive FASTA");
    let neg = read_fasta(positional[1]).expect("reading negative FASTA");
    println!("{} positive / {} negative sequences, length {len}", pos.len(), neg.len());

    let mut seqs: Vec<Vec<u8>> = pos.iter().map(|r| fit_length(&r.1, len)).collect();
    let mut labels = vec![1i64; seqs.len()];
    seqs.extend(neg.iter().map(|r| fit_length(&r.1, len)));
    labels.extend(vec![0i64; neg.len()]);
    let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
    let make = |cf: bool| Dataset::from_sequences(&refs, &labels, &Alphabet::DNA, len, cf).unwrap().split(0.2, seed);
    let (cf_train, cf_val) = make(true);
    let (cl_train, cl_val) = make(false);

    println!("\n| model | accuracy | macro-F1 | MCC | AUROC | AUPRC | train time (s) |");
    println!("|---|---|---|---|---|---|---|");
    for name in models.split(',') {
        let (model, cf, lr) = build(name.trim(), len)?;
        let (train, val) = if cf { (&cf_train, &cf_val) } else { (&cl_train, &cl_val) };
        let mut optimizer = adam(model.as_ref(), lr);
        let t = Instant::now();
        fit(model.as_ref(), &mut optimizer, train, Some(val), &TrainConfig { epochs, batch_size: batch, patience: Some(5), seed, verbose: false, ..Default::default() })?;
        let secs = t.elapsed().as_secs_f64();
        let r = report(model.as_ref(), val, 2, batch)?;
        println!("| {name} | {:.4} | {:.4} | {} | {} | {} | {secs:.1} |", r.accuracy, r.macro_f1, common::opt(r.mcc), common::opt(r.auroc), common::opt(r.auprc));
    }
    Ok(())
}
