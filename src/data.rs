//! Datasets, batching, a seeded RNG and classification metrics. Pure Rust.

use crate::alphabet::Alphabet;
use crate::encode::one_hot_batch;
use crate::{Error, Result};

/*
Gaurav Sablok
gsablok@proton.me
*/

/// xorshift64* generator: tiny, seedable, reproducible.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

/// In-memory classification dataset. `x` is row-major `[n, sample_shape...]`, `y` class indices.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: Vec<f32>,
    pub sample_shape: Vec<usize>,
    pub y: Vec<i64>,
    pub n: usize,
}

impl Dataset {
    pub fn new(x: Vec<f32>, sample_shape: Vec<usize>, y: Vec<i64>) -> Result<Self> {
        let sl: usize = sample_shape.iter().product();
        if sl == 0 || x.len() != sl * y.len() {
            return Err(Error::Shape(format!(
                "x has {} values, expected {} samples x {} = {}",
                x.len(),
                y.len(),
                sl,
                sl * y.len()
            )));
        }
        let n = y.len();
        Ok(Dataset {
            x,
            sample_shape,
            y,
            n,
        })
    }

    pub fn sample_len(&self) -> usize {
        self.sample_shape.iter().product()
    }

    /// One-hot encode sequences (padded/truncated to `len`). `channels_first` gives
    /// `[A, L]` samples (for Conv1d), otherwise `[L, A]` (for RNN / Transformer).
    pub fn from_sequences(
        seqs: &[&[u8]],
        labels: &[i64],
        alpha: &Alphabet,
        len: usize,
        channels_first: bool,
    ) -> Result<Self> {
        if seqs.len() != labels.len() {
            return Err(Error::Shape("seqs and labels differ in length".into()));
        }
        let b = one_hot_batch(seqs, alpha, Some(len));
        let (x, shape) = if channels_first {
            (b.channels_first(), vec![alpha.len(), len])
        } else {
            (b.data, vec![len, alpha.len()])
        };
        Dataset::new(x, shape, labels.to_vec())
    }

    /// Gather a batch: returns flat x values and labels.
    pub fn batch(&self, idx: &[usize]) -> (Vec<f32>, Vec<i64>) {
        let sl = self.sample_len();
        let mut x = Vec::with_capacity(idx.len() * sl);
        let mut y = Vec::with_capacity(idx.len());
        for &i in idx {
            x.extend_from_slice(&self.x[i * sl..(i + 1) * sl]);
            y.push(self.y[i]);
        }
        (x, y)
    }

    /// Random train/validation split.
    pub fn split(&self, val_frac: f32, seed: u64) -> (Dataset, Dataset) {
        let mut idx: Vec<usize> = (0..self.n).collect();
        Rng::new(seed).shuffle(&mut idx);
        let n_val = ((self.n as f32) * val_frac).round() as usize;
        let (val_idx, train_idx) = idx.split_at(n_val.min(self.n));
        let make = |ix: &[usize]| {
            let (x, y) = self.batch(ix);
            Dataset {
                x,
                sample_shape: self.sample_shape.clone(),
                y,
                n: ix.len(),
            }
        };
        (make(train_idx), make(val_idx))
    }
}

/// Mini-batch index lists, optionally shuffled.
pub fn batch_indices(n: usize, batch_size: usize, rng: Option<&mut Rng>) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = (0..n).collect();
    if let Some(r) = rng {
        r.shuffle(&mut idx);
    }
    idx.chunks(batch_size.max(1)).map(|c| c.to_vec()).collect()
}

/// Synthetic motif task for smoke tests: class 1 sequences contain `motif` at a random
/// position inside random DNA; class 0 are random. Balanced classes.
pub fn synthetic_motif(n: usize, len: usize, motif: &[u8], seed: u64) -> (Vec<Vec<u8>>, Vec<i64>) {
    let mut rng = Rng::new(seed);
    let mut seqs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let mut s: Vec<u8> = (0..len).map(|_| b"ACGT"[rng.below(4)]).collect();
        let positive = i % 2 == 1;
        if positive && len >= motif.len() {
            let p = rng.below(len - motif.len() + 1);
            s[p..p + motif.len()].copy_from_slice(motif);
        }
        seqs.push(s);
        labels.push(positive as i64);
    }
    (seqs, labels)
}

// ---------------------------------------------------------------- metrics

pub fn accuracy(pred: &[usize], y: &[i64]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    pred.iter()
        .zip(y)
        .filter(|(p, t)| **p as i64 == **t)
        .count() as f64
        / y.len() as f64
}

/// Row = true class, column = predicted class, row-major `[classes, classes]`.
pub fn confusion_matrix(pred: &[usize], y: &[i64], classes: usize) -> Vec<usize> {
    let mut cm = vec![0usize; classes * classes];
    for (&p, &t) in pred.iter().zip(y) {
        if (t as usize) < classes && p < classes {
            cm[t as usize * classes + p] += 1;
        }
    }
    cm
}

pub fn macro_f1(cm: &[usize], classes: usize) -> f64 {
    let mut sum = 0.0;
    for c in 0..classes {
        let tp = cm[c * classes + c] as f64;
        let fp: f64 = (0..classes)
            .filter(|&r| r != c)
            .map(|r| cm[r * classes + c] as f64)
            .sum();
        let fn_: f64 = (0..classes)
            .filter(|&k| k != c)
            .map(|k| cm[c * classes + k] as f64)
            .sum();
        let denom = 2.0 * tp + fp + fn_;
        sum += if denom > 0.0 { 2.0 * tp / denom } else { 0.0 };
    }
    sum / classes as f64
}

/// Matthews correlation coefficient from a 2x2 confusion matrix.
pub fn mcc_binary(cm: &[usize]) -> f64 {
    let (tn, fp, fn_, tp) = (cm[0] as f64, cm[1] as f64, cm[2] as f64, cm[3] as f64);
    let d = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if d == 0.0 {
        0.0
    } else {
        (tp * tn - fp * fn_) / d
    }
}

/// Binary AUROC via the rank-sum statistic (ties get average ranks). Label 1 = positive.
pub fn auroc(scores: &[f32], labels: &[i64]) -> Option<f64> {
    let n = scores.len();
    let pos = labels.iter().filter(|&&l| l == 1).count();
    let neg = n - pos;
    if pos == 0 || neg == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[order[k]] = avg;
        }
        i = j + 1;
    }
    let rank_sum: f64 = (0..n).filter(|&k| labels[k] == 1).map(|k| ranks[k]).sum();
    Some((rank_sum - pos as f64 * (pos as f64 + 1.0) / 2.0) / (pos as f64 * neg as f64))
}

/// Average precision (area under the precision-recall curve, step-wise). Label 1 = positive.
/// Tied scores are treated as a single threshold.
pub fn average_precision(scores: &[f32], labels: &[i64]) -> Option<f64> {
    let n = scores.len();
    let pos = labels.iter().filter(|&&l| l == 1).count();
    if pos == 0 || pos == n {
        return None;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (mut tp, mut seen, mut ap, mut prev_recall) = (0usize, 0usize, 0.0f64, 0.0f64);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        for k in i..=j {
            if labels[order[k]] == 1 {
                tp += 1;
            }
        }
        seen += j - i + 1;
        let precision = tp as f64 / seen as f64;
        let recall = tp as f64 / pos as f64;
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
        i = j + 1;
    }
    Some(ap)
}

/// Synthetic graph-classification task for demos and benchmarks. Each sample is a one-hot
/// node-feature matrix `[nodes, types]`; positive graphs over-represent type 0 in the first
/// third of nodes. Pair it with a fixed adjacency shared by all samples.
pub fn synthetic_graph_task(n: usize, nodes: usize, types: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let mut x = vec![0.0f32; n * nodes * types];
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let positive = i % 2 == 1;
        y.push(positive as i64);
        for j in 0..nodes {
            let biased = positive && j < nodes / 3 && rng.below(10) < 7;
            let t = if biased { 0 } else { rng.below(types) };
            x[(i * nodes + j) * types + t] = 1.0;
        }
    }
    Dataset {
        x,
        sample_shape: vec![nodes, types],
        y,
        n,
    }
}
