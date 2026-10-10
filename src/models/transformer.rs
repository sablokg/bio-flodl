//! Transformer encoder classifier. Input `[batch, length, alphabet]` (one-hot).

use super::{from_f32, Constant};
use crate::encode::sinusoidal_positions;
use flodl::{Dropout, LayerNorm, Linear, Module, MultiheadAttention, Parameter, Tensor, Variable};

/*
Gaurav Sablok
gsablok@proton.me
*/

pub struct EncoderLayer {
    ln1: LayerNorm,
    attn: MultiheadAttention,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    drop: Dropout,
}

impl EncoderLayer {
    pub fn new(d_model: i64, heads: i64, d_ff: i64, dropout: f64) -> flodl::Result<Self> {
        Ok(EncoderLayer {
            ln1: LayerNorm::new(d_model)?,
            attn: MultiheadAttention::new(d_model, heads)?,
            ln2: LayerNorm::new(d_model)?,
            ff1: Linear::new(d_model, d_ff)?,
            ff2: Linear::new(d_ff, d_model)?,
            drop: Dropout::new(dropout),
        })
    }

    /// Pre-LayerNorm block. `mask` (optional) follows `MultiheadAttention::forward_ext`:
    /// `[seq, seq]` or `[batch, 1, seq, seq]`, non-zero = masked.
    pub fn forward_masked(&self, x: &Variable, mask: Option<&Tensor>) -> flodl::Result<Variable> {
        let h = self.ln1.forward(x)?;
        let a = self.attn.forward_ext(&h, &h, &h, mask)?;
        let x = x.add(&self.drop.forward(&a)?)?;
        let h = self.ln2.forward(&x)?;
        let f = self.ff2.forward(&self.ff1.forward(&h)?.gelu()?)?;
        x.add(&self.drop.forward(&f)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.ln1.parameters();
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.ff1.parameters());
        p.extend(self.ff2.parameters());
        p
    }

    fn set_training(&self, t: bool) {
        self.drop.set_training(t);
    }
}

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub alphabet: i64,
    pub d_model: i64,
    pub heads: i64,
    pub layers: usize,
    pub d_ff: i64,
    pub max_len: usize,
    pub dropout: f64,
    pub classes: i64,
}

pub struct TransformerClassifier {
    embed: Linear,
    pos: Constant,
    layers: Vec<EncoderLayer>,
    ln_f: LayerNorm,
    drop: Dropout,
    head: Linear,
}

impl TransformerClassifier {
    pub fn new(cfg: &TransformerConfig) -> flodl::Result<Self> {
        assert!(
            cfg.d_model % cfg.heads == 0,
            "d_model must be divisible by heads"
        );
        let pe = sinusoidal_positions(cfg.max_len, cfg.d_model as usize);
        let pos = Constant::new(Variable::new(
            from_f32(&pe, &[1, cfg.max_len as i64, cfg.d_model])?,
            false,
        ));
        let mut layers = Vec::new();
        for _ in 0..cfg.layers {
            layers.push(EncoderLayer::new(
                cfg.d_model,
                cfg.heads,
                cfg.d_ff,
                cfg.dropout,
            )?);
        }
        Ok(TransformerClassifier {
            embed: Linear::new(cfg.alphabet, cfg.d_model)?,
            pos,
            layers,
            ln_f: LayerNorm::new(cfg.d_model)?,
            drop: Dropout::new(cfg.dropout),
            head: Linear::new(cfg.d_model, cfg.classes)?,
        })
    }

    /// Per-position contextual embeddings `[batch, length, d_model]`.
    pub fn encode(&self, x: &Variable, mask: Option<&Tensor>) -> flodl::Result<Variable> {
        let l = x.shape()[1];
        let pos = self.pos.on(x.device())?.narrow(1, 0, l)?;
        let mut h = self.embed.forward(x)?.add(&pos)?;
        h = self.drop.forward(&h)?;
        for layer in &self.layers {
            h = layer.forward_masked(&h, mask)?;
        }
        self.ln_f.forward(&h)
    }
}

impl Module for TransformerClassifier {
    fn name(&self) -> &str {
        "transformer_classifier"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let pooled = self.encode(x, None)?.mean_dim(1, false)?;
        self.head.forward(&pooled)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.embed.parameters();
        for l in &self.layers {
            p.extend(l.parameters());
        }
        p.extend(self.ln_f.parameters());
        p.extend(self.head.parameters());
        p
    }

    fn set_training(&self, t: bool) {
        self.drop.set_training(t);
        for l in &self.layers {
            l.set_training(t);
        }
    }
}
