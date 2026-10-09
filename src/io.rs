//! Minimal sequence I/O.

use std::path::Path;

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Parse FASTA text into `(header, uppercase sequence)` records. Blank lines are ignored and
/// multi-line sequences are joined.
pub fn parse_fasta(text: &str) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(h) = line.strip_prefix('>') {
            out.push((h.trim().to_string(), Vec::new()));
        } else if let Some(last) = out.last_mut() {
            last.1.extend(line.bytes().map(|b| b.to_ascii_uppercase()));
        }
    }
    out
}

pub fn read_fasta<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<(String, Vec<u8>)>> {
    Ok(parse_fasta(&std::fs::read_to_string(path)?))
}

/// Centre-crop or right/left-pad (with `N`) to exactly `len` bases.
pub fn fit_length(seq: &[u8], len: usize) -> Vec<u8> {
    if seq.len() >= len {
        let start = (seq.len() - len) / 2;
        seq[start..start + len].to_vec()
    } else {
        let mut v = seq.to_vec();
        v.resize(len, b'N');
        v
    }
}
