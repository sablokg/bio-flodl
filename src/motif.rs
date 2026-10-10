//! Motifs from convolutional filters, the DeepBind / Basset way: a filter's motif is the
//! average of the sequence windows that activate it strongly, written in MEME format for
//! comparison against databases such as JASPAR with TOMTOM.
//!
//! Sequences are one-hot and channels-first, `[n, alphabet, len]`, the layout
//! [`crate::models::cnn::MotifCnn`] takes; activations are `[n, filters, len - width + 1]`,
//! what [`crate::models::cnn::MotifCnn::scan`] returns for an unpadded convolution.

use std::fmt::Write as _;

/// Position frequency matrix: `width` rows of `alphabet` letter probabilities, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Pfm {
    pub width: usize,
    pub alphabet: usize,
    pub probs: Vec<f32>,
    /// Number of sequence windows averaged into the matrix; 0 for a filter nothing activated.
    pub sites: usize,
}

impl Pfm {
    /// Most probable letter at each position.
    pub fn consensus(&self, letters: &[u8]) -> String {
        self.probs
            .chunks(self.alphabet)
            .map(|row| {
                let best =
                    row.iter()
                        .enumerate()
                        .fold((0, f32::MIN), |b, (i, &p)| if p > b.1 { (i, p) } else { b });
                letters[best.0] as char
            })
            .collect()
    }
}

/// One matrix per filter. A window counts when the filter's activation there exceeds
/// `threshold` times the filter's largest activation over all sequences (DeepBind and Basset
/// use 0.5); counted windows are summed per position and each row is normalised, so a row
/// covering an unknown base (an all-zero one-hot column) is normalised over the bases seen.
#[allow(clippy::too_many_arguments)]
pub fn pfms_from_activations(
    x: &[f32],
    n: usize,
    alphabet: usize,
    len: usize,
    acts: &[f32],
    filters: usize,
    width: usize,
    threshold: f32,
) -> Vec<Pfm> {
    let positions = len + 1 - width;
    assert_eq!(x.len(), n * alphabet * len, "x is not [n, alphabet, len]");
    assert_eq!(
        acts.len(),
        n * filters * positions,
        "activations are not [n, filters, len - width + 1]"
    );
    let act = |s: usize, f: usize, p: usize| acts[(s * filters + f) * positions + p];
    (0..filters)
        .map(|f| {
            let max = (0..n)
                .flat_map(|s| (0..positions).map(move |p| (s, p)))
                .map(|(s, p)| act(s, f, p))
                .fold(f32::MIN, f32::max);
            let mut counts = vec![0.0f32; width * alphabet];
            let mut sites = 0;
            if max > 0.0 {
                for s in 0..n {
                    for p in 0..positions {
                        if act(s, f, p) > threshold * max {
                            sites += 1;
                            for w in 0..width {
                                for a in 0..alphabet {
                                    counts[w * alphabet + a] += x[(s * alphabet + a) * len + p + w];
                                }
                            }
                        }
                    }
                }
            }
            for row in counts.chunks_mut(alphabet) {
                let total: f32 = row.iter().sum();
                if total > 0.0 {
                    row.iter_mut().for_each(|c| *c /= total);
                }
            }
            Pfm {
                width,
                alphabet,
                probs: counts,
                sites,
            }
        })
        .collect()
}

/// Letter frequencies over every position of `x` (`[n, alphabet, len]`), for the MEME
/// background; uniform when `x` holds no known base.
pub fn background(x: &[f32], n: usize, alphabet: usize, len: usize) -> Vec<f64> {
    let mut freq = vec![0.0f64; alphabet];
    for s in 0..n {
        for (a, f) in freq.iter_mut().enumerate() {
            let start = (s * alphabet + a) * len;
            *f += x[start..start + len].iter().map(|&v| v as f64).sum::<f64>();
        }
    }
    let total: f64 = freq.iter().sum();
    if total == 0.0 {
        return vec![1.0 / alphabet as f64; alphabet];
    }
    freq.iter().map(|f| f / total).collect()
}

/// MEME minimal motif format (version 4), as TOMTOM reads it. `letters` are the alphabet's
/// symbols in one-hot channel order (`Alphabet::DNA.symbols()`); motifs with no sites are
/// left out.
pub fn write_meme(motifs: &[(String, Pfm)], letters: &[u8], background: &[f64]) -> String {
    let mut out = String::from("MEME version 4\n\n");
    let _ = writeln!(out, "ALPHABET= {}\n", String::from_utf8_lossy(letters));
    out.push_str("strands: + -\n\nBackground letter frequencies\n");
    let bg: Vec<String> = letters
        .iter()
        .zip(background)
        .map(|(&l, f)| format!("{} {f:.4}", l as char))
        .collect();
    let _ = writeln!(out, "{}\n", bg.join(" "));
    for (name, pfm) in motifs.iter().filter(|(_, p)| p.sites > 0) {
        let _ = writeln!(out, "MOTIF {name}");
        let _ = writeln!(
            out,
            "letter-probability matrix: alength= {} w= {} nsites= {} E= 0",
            pfm.alphabet, pfm.width, pfm.sites
        );
        for row in pfm.probs.chunks(pfm.alphabet) {
            let cells: Vec<String> = row.iter().map(|p| format!("{p:.4}")).collect();
            let _ = writeln!(out, " {}", cells.join(" "));
        }
        out.push('\n');
    }
    out
}
