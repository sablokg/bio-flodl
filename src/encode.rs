//! Sequence encodings for machine learning.

use crate::alphabet::Alphabet;
use crate::{Error, Result};

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Row-major `[len, width]` one-hot matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct OneHot {
    pub data: Vec<f32>,
    pub len: usize,
    pub width: usize,
}

impl OneHot {
    pub fn shape(&self) -> [usize; 2] {
        [self.len, self.width]
    }

    /// Transpose to channels-first `[width, len]`, the layout 1-D convolutions expect.
    pub fn channels_first(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.data.len()];
        for l in 0..self.len {
            for c in 0..self.width {
                out[c * self.len + l] = self.data[l * self.width + c];
            }
        }
        out
    }
}

/// One-hot encode a sequence; unknown symbols are an error.
pub fn one_hot(seq: &[u8], alpha: &Alphabet) -> Result<OneHot> {
    let idx = alpha.encode(seq)?;
    Ok(from_indices(&idx, alpha.len()))
}

/// One-hot encode, mapping unknown symbols (e.g. `N`, `X`) to an all-zero row.
pub fn one_hot_lenient(seq: &[u8], alpha: &Alphabet) -> OneHot {
    let w = alpha.len();
    let mut data = vec![0.0; seq.len() * w];
    for (i, &b) in seq.iter().enumerate() {
        if let Some(j) = alpha.index(b) {
            data[i * w + j] = 1.0;
        }
    }
    OneHot {
        data,
        len: seq.len(),
        width: w,
    }
}

/// One-hot from class indices.
pub fn from_indices(idx: &[usize], width: usize) -> OneHot {
    let mut data = vec![0.0; idx.len() * width];
    for (i, &j) in idx.iter().enumerate() {
        data[i * width + j] = 1.0;
    }
    OneHot {
        data,
        len: idx.len(),
        width,
    }
}

/// One-hot class labels `[n, classes]` (for classification targets).
pub fn one_hot_labels(labels: &[usize], classes: usize) -> Result<OneHot> {
    if let Some(&bad) = labels.iter().find(|&&l| l >= classes) {
        return Err(Error::Invalid(format!("label {bad} >= classes {classes}")));
    }
    Ok(from_indices(labels, classes))
}

/// Batch of one-hot sequences, zero-padded/truncated to `pad_to` (or the longest sequence).
/// Returns flat `[n, L, width]` data plus a `[n, L]` mask (1 = real residue).
pub struct Batch {
    pub data: Vec<f32>,
    pub mask: Vec<f32>,
    pub n: usize,
    pub len: usize,
    pub width: usize,
}

pub fn one_hot_batch(seqs: &[&[u8]], alpha: &Alphabet, pad_to: Option<usize>) -> Batch {
    let w = alpha.len();
    let l = pad_to.unwrap_or_else(|| seqs.iter().map(|s| s.len()).max().unwrap_or(0));
    let mut data = vec![0.0; seqs.len() * l * w];
    let mut mask = vec![0.0; seqs.len() * l];
    for (n, s) in seqs.iter().enumerate() {
        for (i, &b) in s.iter().take(l).enumerate() {
            mask[n * l + i] = 1.0;
            if let Some(j) = alpha.index(b) {
                data[(n * l + i) * w + j] = 1.0;
            }
        }
    }
    Batch {
        data,
        mask,
        n: seqs.len(),
        len: l,
        width: w,
    }
}

impl Batch {
    pub fn shape(&self) -> [usize; 3] {
        [self.n, self.len, self.width]
    }

    /// `[n, width, L]` channels-first layout for convolutions.
    pub fn channels_first(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.data.len()];
        for n in 0..self.n {
            for l in 0..self.len {
                for c in 0..self.width {
                    out[(n * self.width + c) * self.len + l] =
                        self.data[(n * self.len + l) * self.width + c];
                }
            }
        }
        out
    }
}

/// Index of a k-mer in base-`A` positional notation, or `None` if it has unknown symbols.
pub fn kmer_index(kmer: &[u8], alpha: &Alphabet) -> Option<usize> {
    kmer.iter().try_fold(0usize, |acc, &b| {
        alpha.index(b).map(|j| acc * alpha.len() + j)
    })
}

/// k-mer count vector of length `A^k` (windows with unknown symbols are skipped).
pub fn kmer_counts(seq: &[u8], k: usize, alpha: &Alphabet) -> Result<Vec<f32>> {
    if k == 0 {
        return Err(Error::Invalid("k must be > 0".into()));
    }
    let size = alpha
        .len()
        .checked_pow(k as u32)
        .ok_or_else(|| Error::Invalid("k too large".into()))?;
    let mut v = vec![0.0; size];
    for w in seq.windows(k) {
        if let Some(i) = kmer_index(w, alpha) {
            v[i] += 1.0;
        }
    }
    Ok(v)
}

/// Frequencies (counts / total windows counted).
pub fn kmer_frequencies(seq: &[u8], k: usize, alpha: &Alphabet) -> Result<Vec<f32>> {
    let mut v = kmer_counts(seq, k, alpha)?;
    let total: f32 = v.iter().sum();
    if total > 0.0 {
        v.iter_mut().for_each(|x| *x /= total);
    }
    Ok(v)
}

/// Sinusoidal positional encodings (Vaswani et al.), row-major `[max_len, d_model]`.
pub fn sinusoidal_positions(max_len: usize, d_model: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; max_len * d_model];
    for pos in 0..max_len {
        for i in 0..d_model {
            let pair = (i / 2) as f32;
            let angle = pos as f32 / 10000f32.powf(2.0 * pair / d_model as f32);
            pe[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    pe
}
