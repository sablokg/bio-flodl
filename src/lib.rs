//! # bio-flodl
//!
//! Bioinformatics data layer for [flodl](https://docs.rs/flodl).
//!
//! * [`alphabet`] – DNA / RNA / protein alphabets
//! * [`encode`]   – one-hot, label, k-mer encodings, batching/padding, channels-first for conv
//! * [`graph`]    – graphs, dense adjacency, GCN-normalised adjacency, Laplacian,
//!                  contact maps, de Bruijn graphs, connected components
//! * [`io`]       – FASTA parsing, fixed-length cropping/padding
//! * [`data`]     – datasets, batching, seeded RNG, metrics (accuracy, F1, MCC, AUROC)
//! * `bridge` (feature `flodl`) – conversion to flodl tensors
//! * `models` (feature `flodl`) – motif CNNs, BiLSTM/BiGRU, Transformer, GCN, GAT
//! * `train` (feature `flodl`) – training / evaluation loops
//! * `prelude` (feature `flodl`) – all of flodl + everything above in one import
//!
//! The core is pure Rust with no libtorch dependency, so it builds and tests anywhere.

pub mod alphabet;
pub mod data;
pub mod encode;
pub mod graph;
pub mod io;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[cfg(feature = "flodl")]
pub mod bridge;
#[cfg(feature = "flodl")]
pub mod models;
#[cfg(feature = "flodl")]
pub mod train;

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    UnknownSymbol { symbol: char, position: usize },
    Shape(String),
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnknownSymbol { symbol, position } => {
                write!(
                    f,
                    "symbol '{symbol}' at position {position} not in alphabet"
                )
            }
            Error::Shape(s) => write!(f, "shape error: {s}"),
            Error::Invalid(s) => write!(f, "invalid argument: {s}"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

/// One import for everything: **all of flodl** (layers, losses, optimizers, schedulers, tensors,
/// graph builder, data loaders, ...) plus this crate's models, containers and training loop.
///
/// Names defined by bio-flodl take precedence over flodl's on collision (they are explicit
/// re-exports; flodl's come through the glob).
#[cfg(feature = "flodl")]
pub mod prelude {
    pub use flodl::*;

    pub use crate::models::cnn::{DilatedCnnConfig, DilatedResCnn, MotifCnn, MotifCnnConfig};
    pub use crate::models::compose::{ConcatBranches, Lambda, Residual, Sequential};
    pub use crate::models::gnn::{GatLayer, GatNet, GcnLayer, GcnNet, Readout};
    pub use crate::models::mlp::FlatMlp;
    pub use crate::models::rnn::{BiRnnClassifier, RnnKind};
    pub use crate::models::transformer::{EncoderLayer, TransformerClassifier, TransformerConfig};
    pub use crate::train::{
        adam, evaluate, fit, fit_with, predict_proba, predict_probs, report, EpochStats, Report,
        TrainConfig,
    };
}
