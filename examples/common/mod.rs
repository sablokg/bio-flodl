//! Helpers shared by the flodl examples (`mod common;`).
#![allow(dead_code)]

use flodl_bio::alphabet::Alphabet;
use flodl_bio::data::{synthetic_motif, Dataset};
use flodl_bio::train::Report;

/// Planted-motif task: DNA of length `len`, positives contain `TATAAA`.
/// `channels_first = true` gives `[4, L]` samples (CNNs), otherwise `[L, 4]` (RNN / Transformer).
/// The split is deterministic, so both layouts yield the same train/validation membership.
pub fn motif_task(n: usize, len: usize, channels_first: bool, seed: u64) -> (Dataset, Dataset) {
    let (seqs, labels) = synthetic_motif(n, len, b"TATAAA", seed);
    let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
    Dataset::from_sequences(&refs, &labels, &Alphabet::DNA, len, channels_first).unwrap().split(0.2, 1)
}

pub fn opt(v: Option<f64>) -> String { v.map_or("   -  ".into(), |x| format!("{x:.4}")) }

pub fn print_report(name: &str, r: &Report) {
    println!("{name}: accuracy {:.4}  macro-F1 {:.4}  MCC {}  AUROC {}  AUPRC {}",
        r.accuracy, r.macro_f1, opt(r.mcc), opt(r.auroc), opt(r.auprc));
}
