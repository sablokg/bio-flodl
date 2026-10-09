//! Bidirectional recurrent classifiers. Input `[batch, length, alphabet]`.

use flodl::{Dropout, Linear, Module, Parameter, Tensor, Variable, GRU, LSTM};

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnnKind {
    Lstm,
    Gru,
}

enum Rnn {
    Lstm(LSTM),
    Gru(GRU),
}

impl Rnn {
    fn new(kind: RnnKind, input: i64, hidden: i64, layers: usize) -> flodl::Result<Self> {
        Ok(match kind {
            RnnKind::Lstm => Rnn::Lstm(LSTM::new(input, hidden, layers)?.batch_first(true)),
            RnnKind::Gru => Rnn::Gru(GRU::new(input, hidden, layers)?.batch_first(true)),
        })
    }
    fn run(&self, x: &Variable) -> flodl::Result<Variable> {
        match self {
            Rnn::Lstm(m) => m.forward(x),
            Rnn::Gru(m) => m.forward(x),
        }
    }
    fn params(&self) -> Vec<Parameter> {
        match self {
            Rnn::Lstm(m) => m.parameters(),
            Rnn::Gru(m) => m.parameters(),
        }
    }
}

/// Forward RNN + RNN over the reversed sequence, outputs concatenated and mean-pooled over
/// time, then dropout and a linear head. (flodl's LSTM/GRU are unidirectional, so the
/// backward direction is a second RNN run on the time-reversed input.)
pub struct BiRnnClassifier {
    fwd: Rnn,
    bwd: Rnn,
    drop: Dropout,
    head: Linear,
}

impl BiRnnClassifier {
    pub fn new(
        kind: RnnKind,
        input: i64,
        hidden: i64,
        layers: usize,
        dropout: f64,
        classes: i64,
    ) -> flodl::Result<Self> {
        Ok(BiRnnClassifier {
            fwd: Rnn::new(kind, input, hidden, layers)?,
            bwd: Rnn::new(kind, input, hidden, layers)?,
            drop: Dropout::new(dropout),
            head: Linear::new(2 * hidden, classes)?,
        })
    }
}

impl Module for BiRnnClassifier {
    fn name(&self) -> &str {
        "bi_rnn"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let l = x.shape()[1];
        let rev: Vec<i64> = (0..l).rev().collect();
        let rev_idx = Tensor::from_i64(&rev, &[l], x.device())?;
        let x_rev = x.index_select(1, &rev_idx)?;
        let of = self.fwd.run(x)?;
        let ob = self.bwd.run(&x_rev)?;
        let pooled = Variable::cat_many(&[&of, &ob], 2)?.mean_dim(1, false)?;
        self.head.forward(&self.drop.forward(&pooled)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.fwd.params();
        p.extend(self.bwd.params());
        p.extend(self.head.parameters());
        p
    }

    fn set_training(&self, t: bool) {
        self.drop.set_training(t);
    }
}
