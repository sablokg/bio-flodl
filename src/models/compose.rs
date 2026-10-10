//! Generic containers so *any* flodl layer (Conv2d, Conv3d, ConvTranspose*, BatchNorm, GroupNorm,
//! RMSNorm, MaxPool1d, AvgPool1d, Embedding, EmbeddingBag, GRU, LSTM, MultiheadAttention, ...)
//! can be assembled into a model. You construct the layers with flodl's own constructors;
//! these containers wire them together and make them trainable with [`crate::train`].
//!
//! ```ignore
//! use flodl_bio::prelude::*;
//! let model = Sequential::new()
//!     .push(Conv1d::new(4, 64, 9)?)
//!     .push(Lambda::relu())
//!     .push(Dropout::new(0.2))
//!     .push(Lambda::global_max_pool1d())
//!     .push(Linear::new(64, 2)?);
//! ```

use flodl::{Module, Parameter, Variable};
use std::rc::Rc;

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Runs modules one after another.
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Rc<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Append any flodl module (or any of this crate's models).
    pub fn push<M: Module + 'static>(mut self, m: M) -> Self {
        self.layers.push(Rc::new(m));
        self
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn name(&self) -> &str {
        "sequential"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let mut h = x.clone();
        for l in &self.layers {
            h = l.forward(&h)?;
        }
        Ok(h)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        self.layers.clone()
    }

    fn set_training(&self, training: bool) {
        for l in &self.layers {
            l.set_training(training);
        }
    }
}

/// `x + f(x)`. `f` must preserve the shape.
pub struct Residual<M: Module> {
    inner: Rc<M>,
}

impl<M: Module> Residual<M> {
    pub fn new(inner: M) -> Self {
        Residual {
            inner: Rc::new(inner),
        }
    }
}

impl<M: Module + 'static> Module for Residual<M> {
    fn name(&self) -> &str {
        "residual"
    }
    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        x.add(&self.inner.forward(x)?)
    }
    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.inner.clone()]
    }
    fn set_training(&self, t: bool) {
        self.inner.set_training(t);
    }
}

/// Parameter-free function as a layer (activations, pooling, reshapes).
pub struct Lambda {
    name: &'static str,
    f: Box<dyn Fn(&Variable) -> flodl::Result<Variable>>,
}

impl Lambda {
    pub fn new(
        name: &'static str,
        f: impl Fn(&Variable) -> flodl::Result<Variable> + 'static,
    ) -> Self {
        Lambda {
            name,
            f: Box::new(f),
        }
    }

    pub fn relu() -> Self {
        Lambda::new("relu", |x| x.relu())
    }
    pub fn gelu() -> Self {
        Lambda::new("gelu", |x| x.gelu())
    }
    pub fn tanh() -> Self {
        Lambda::new("tanh", |x| x.tanh())
    }
    pub fn sigmoid() -> Self {
        Lambda::new("sigmoid", |x| x.sigmoid())
    }

    /// `[B, C, L] -> [B, C]` (max over the last axis).
    pub fn global_max_pool1d() -> Self {
        Lambda::new("global_max_pool1d", |x| {
            x.max_dim(x.shape().len() as i32 - 1, false)
        })
    }
    /// `[B, C, L] -> [B, C]` (mean over the last axis).
    pub fn global_avg_pool1d() -> Self {
        Lambda::new("global_avg_pool1d", |x| {
            x.mean_dim(x.shape().len() as i32 - 1, false)
        })
    }
    /// `[B, ...] -> [B, prod(...)]`.
    pub fn flatten() -> Self {
        Lambda::new("flatten", |x| {
            let b = x.shape()[0];
            x.reshape(&[b, -1])
        })
    }
    /// Swap two axes, e.g. `Lambda::transpose(1, 2)` converts `[B, C, L]` <-> `[B, L, C]`
    /// (to feed conv features into an LSTM/Transformer or one-hot into a Conv1d).
    pub fn transpose(a: i32, b: i32) -> Self {
        Lambda::new("transpose", move |x| x.transpose(a, b))
    }
}

impl Module for Lambda {
    fn name(&self) -> &str {
        self.name
    }
    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        (self.f)(x)
    }
    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }
}

/// Runs several branches on the same input and concatenates their outputs along `dim`
/// (inception-style multi-width motif scanners, multi-modal heads, ...).
pub struct ConcatBranches {
    branches: Vec<Rc<dyn Module>>,
    dim: i32,
}

impl ConcatBranches {
    pub fn new(dim: i32) -> Self {
        ConcatBranches {
            branches: Vec::new(),
            dim,
        }
    }

    pub fn branch<M: Module + 'static>(mut self, m: M) -> Self {
        self.branches.push(Rc::new(m));
        self
    }
}

impl Module for ConcatBranches {
    fn name(&self) -> &str {
        "concat_branches"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let outs: Vec<Variable> = self
            .branches
            .iter()
            .map(|b| b.forward(x))
            .collect::<flodl::Result<_>>()?;
        let refs: Vec<&Variable> = outs.iter().collect();
        Variable::cat_many(&refs, self.dim)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        self.branches.clone()
    }
    fn set_training(&self, t: bool) {
        for b in &self.branches {
            b.set_training(t);
        }
    }
}
