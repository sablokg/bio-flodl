//! 1-D convolutional models for sequences. Input `[batch, alphabet, length]`.

use flodl::{Conv1d, Device, Dropout, Linear, Module, Parameter, Variable};

// ------------------------------------------------------------------ motif CNN

#[derive(Debug, Clone)]
pub struct MotifCnnConfig {
    pub in_channels: i64,
    /// Filters per kernel size.
    pub filters: i64,
    /// Parallel motif scanners, e.g. `[8, 12, 16]` (DeepBind / DeepSEA style).
    pub kernel_sizes: Vec<i64>,
    pub dropout: f64,
    pub classes: i64,
}

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Parallel Conv1d motif detectors -> ReLU -> global max-pool -> concat -> dropout -> linear.
/// Works for any sequence length >= the largest kernel.
pub struct MotifCnn {
    convs: Vec<Conv1d>,
    drop: Dropout,
    head: Linear,
}

impl MotifCnn {
    pub fn new(cfg: &MotifCnnConfig) -> flodl::Result<Self> {
        let mut convs = Vec::new();
        for &k in &cfg.kernel_sizes {
            convs.push(Conv1d::new(cfg.in_channels, cfg.filters, k)?);
        }
        let head = Linear::new(cfg.filters * cfg.kernel_sizes.len() as i64, cfg.classes)?;
        Ok(MotifCnn {
            convs,
            drop: Dropout::new(cfg.dropout),
            head,
        })
    }

    /// Learned motif filters of scanner `i`: `[filters, alphabet, kernel]`.
    /// Interpret each filter as a position weight matrix over the alphabet.
    pub fn motif_filters(&self, i: usize) -> Option<Variable> {
        self.convs.get(i).map(|c| c.weight.variable.clone())
    }
}

impl Module for MotifCnn {
    fn name(&self) -> &str {
        "motif_cnn"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let mut pooled = Vec::with_capacity(self.convs.len());
        for c in &self.convs {
            pooled.push(c.forward(x)?.relu()?.max_dim(2, false)?);
        }
        let refs: Vec<&Variable> = pooled.iter().collect();
        let h = Variable::cat_many(&refs, 1)?;
        self.head.forward(&self.drop.forward(&h)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p: Vec<Parameter> = self.convs.iter().flat_map(|c| c.parameters()).collect();
        p.extend(self.head.parameters());
        p
    }

    fn set_training(&self, t: bool) {
        self.drop.set_training(t);
    }
}

// -------------------------------------------------------- dilated residual CNN

#[derive(Debug, Clone)]
pub struct DilatedCnnConfig {
    pub in_channels: i64,
    pub channels: i64,
    /// Odd kernel size (so "same" padding is exact).
    pub kernel: i64,
    /// Residual blocks; block `i` uses dilation `2^i`, so the receptive field grows exponentially.
    pub blocks: usize,
    pub dropout: f64,
    pub classes: i64,
}

/// Stem conv + residual dilated conv blocks + (mean ‖ max) global pooling + linear head.
/// Length-preserving, so it also suits per-position tasks if you swap the head.
pub struct DilatedResCnn {
    stem: Conv1d,
    blocks: Vec<Conv1d>,
    drop: Dropout,
    head: Linear,
}

impl DilatedResCnn {
    pub fn new(cfg: &DilatedCnnConfig) -> flodl::Result<Self> {
        assert!(cfg.kernel % 2 == 1, "kernel must be odd for same-padding");
        let k = cfg.kernel;
        let stem = Conv1d::build(
            cfg.in_channels,
            cfg.channels,
            k,
            true,
            1,
            k / 2,
            1,
            1,
            Device::CPU,
        )?;
        let mut blocks = Vec::new();
        for i in 0..cfg.blocks {
            let d = 1i64 << i;
            blocks.push(Conv1d::build(
                cfg.channels,
                cfg.channels,
                k,
                true,
                1,
                d * (k - 1) / 2,
                d,
                1,
                Device::CPU,
            )?);
        }
        let head = Linear::new(2 * cfg.channels, cfg.classes)?;
        Ok(DilatedResCnn {
            stem,
            blocks,
            drop: Dropout::new(cfg.dropout),
            head,
        })
    }

    /// Per-position features `[batch, channels, length]` (before pooling).
    pub fn features(&self, x: &Variable) -> flodl::Result<Variable> {
        let mut h = self.stem.forward(x)?.relu()?;
        for b in &self.blocks {
            h = h.add(&b.forward(&h)?.relu()?)?;
        }
        Ok(h)
    }
}

impl Module for DilatedResCnn {
    fn name(&self) -> &str {
        "dilated_res_cnn"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let h = self.features(x)?;
        let pooled = Variable::cat_many(&[&h.mean_dim(2, false)?, &h.max_dim(2, false)?], 1)?;
        self.head.forward(&self.drop.forward(&pooled)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.stem.parameters();
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p.extend(self.head.parameters());
        p
    }

    fn set_training(&self, t: bool) {
        self.drop.set_training(t);
    }
}
