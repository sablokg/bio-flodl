//! Biological alphabets. Lookup is case-insensitive.

use crate::{Error, Result};

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone, Copy)]
pub struct Alphabet {
    pub name: &'static str,
    symbols: &'static [u8],
}

impl Alphabet {
    pub const DNA: Alphabet = Alphabet {
        name: "DNA",
        symbols: b"ACGT",
    };
    pub const RNA: Alphabet = Alphabet {
        name: "RNA",
        symbols: b"ACGU",
    };
    /// 20 standard amino acids.
    pub const PROTEIN: Alphabet = Alphabet {
        name: "PROTEIN",
        symbols: b"ACDEFGHIKLMNPQRSTVWY",
    };

    pub const fn len(&self) -> usize {
        self.symbols.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
    pub fn symbols(&self) -> &'static [u8] {
        self.symbols
    }

    pub fn index(&self, b: u8) -> Option<usize> {
        let u = b.to_ascii_uppercase();
        self.symbols.iter().position(|&s| s == u)
    }

    pub fn symbol(&self, idx: usize) -> Option<char> {
        self.symbols.get(idx).map(|&b| b as char)
    }

    /// Label-encode a sequence; errors on the first unknown symbol.
    pub fn encode(&self, seq: &[u8]) -> Result<Vec<usize>> {
        seq.iter()
            .enumerate()
            .map(|(i, &b)| {
                self.index(b).ok_or(Error::UnknownSymbol {
                    symbol: b as char,
                    position: i,
                })
            })
            .collect()
    }

    pub fn decode(&self, idx: &[usize]) -> Result<String> {
        idx.iter()
            .map(|&i| {
                self.symbol(i)
                    .ok_or_else(|| Error::Invalid(format!("index {i} out of range")))
            })
            .collect()
    }
}

/// Complement of a DNA base (case preserved); other bytes are returned as `N`.
pub fn complement_dna(b: u8) -> u8 {
    match b {
        b'A' => b'T',
        b'T' => b'A',
        b'C' => b'G',
        b'G' => b'C',
        b'a' => b't',
        b't' => b'a',
        b'c' => b'g',
        b'g' => b'c',
        _ => b'N',
    }
}

pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| complement_dna(b)).collect()
}

pub fn gc_content(seq: &[u8]) -> f32 {
    if seq.is_empty() {
        return 0.0;
    }
    let gc = seq
        .iter()
        .filter(|b| matches!(b.to_ascii_uppercase(), b'G' | b'C'))
        .count();
    gc as f32 / seq.len() as f32
}
