//! Graph neural networks on dense adjacency matrices.
//!
//! Node features are `[nodes, features]` or `[batch, nodes, features]` (shared adjacency).
//! Dense message passing is O(N²) and suits molecules, small contact graphs and
//! residue graphs; it is not intended for graphs with 100k+ nodes.

use super::{glorot, zeros_param};
use flodl::{Dropout, Linear, Module, Parameter, Variable};

/*
Gaurav Sablok
gsablok@proton.me
*/

// ------------------------------------------------------------------------ GCN

/// `H' = Â H W + b` with `Â = D^-1/2 (A+I) D^-1/2` (Kipf & Welling).
pub struct GcnLayer {
    w: Parameter,
    b: Parameter,
}

impl GcnLayer {
    pub fn new(in_f: i64, out_f: i64) -> flodl::Result<Self> {
        Ok(GcnLayer {
            w: glorot(&[in_f, out_f], in_f, out_f, "gcn.w")?,
            b: zeros_param(&[out_f], "gcn.b")?,
        })
    }

    pub fn forward_with(&self, x: &Variable, a_hat: &Variable) -> flodl::Result<Variable> {
        let h = x.matmul(&self.w.variable)?;
        a_hat.matmul(&h)?.add(&self.b.variable)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.w.clone(), self.b.clone()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    /// One prediction per node (node classification).
    Node,
    /// Mean over nodes, then one prediction per graph (graph classification).
    GraphMean,
}

pub struct GcnNet {
    layers: Vec<GcnLayer>,
    a_hat: Variable,
    drop: Dropout,
    head: Linear,
    readout: Readout,
}

impl GcnNet {
    /// `a_hat`: normalised adjacency, e.g. [`crate::bridge::gcn_adjacency_variable`].
    /// Create it on the device the model will run on.
    pub fn new(
        a_hat: Variable,
        in_f: i64,
        hidden: i64,
        depth: usize,
        classes: i64,
        dropout: f64,
        readout: Readout,
    ) -> flodl::Result<Self> {
        assert!(depth >= 1);
        let mut layers = vec![GcnLayer::new(in_f, hidden)?];
        for _ in 1..depth {
            layers.push(GcnLayer::new(hidden, hidden)?);
        }
        Ok(GcnNet {
            layers,
            a_hat,
            drop: Dropout::new(dropout),
            head: Linear::new(hidden, classes)?,
            readout,
        })
    }
}

impl Module for GcnNet {
    fn name(&self) -> &str {
        "gcn_net"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let mut h = x.clone();
        for l in &self.layers {
            h = self
                .drop
                .forward(&l.forward_with(&h, &self.a_hat)?.relu()?)?;
        }
        if self.readout == Readout::GraphMean {
            let node_dim = h.shape().len() as i32 - 2;
            h = h.mean_dim(node_dim, false)?;
        }
        self.head.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p: Vec<Parameter> = self.layers.iter().flat_map(|l| l.parameters()).collect();
        p.extend(self.head.parameters());
        p
    }

    fn set_training(&self, t: bool) {
        self.drop.set_training(t);
    }
}

// ------------------------------------------------------------------------ GAT

struct GatHead {
    w: Parameter,
    a_src: Parameter,
    a_dst: Parameter,
}

impl GatHead {
    fn new(in_f: i64, out_f: i64) -> flodl::Result<Self> {
        Ok(GatHead {
            w: glorot(&[in_f, out_f], in_f, out_f, "gat.w")?,
            a_src: glorot(&[out_f, 1], out_f, 1, "gat.a_src")?,
            a_dst: glorot(&[out_f, 1], out_f, 1, "gat.a_dst")?,
        })
    }

    /// `e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j)`, masked to existing edges, softmax over j.
    fn forward_with(&self, x: &Variable, neg_mask: &Variable) -> flodl::Result<Variable> {
        let h = x.matmul(&self.w.variable)?;
        let s = h.matmul(&self.a_src.variable)?;
        let d = h.matmul(&self.a_dst.variable)?;
        let r = s.shape().len() as i32;
        let e = s
            .add(&d.transpose(r - 1, r - 2)?)?
            .leaky_relu(0.2)?
            .add(neg_mask)?;
        e.softmax(r - 1)?.matmul(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.w.clone(), self.a_src.clone(), self.a_dst.clone()]
    }
}

/// Multi-head graph attention (Veličković et al.). Heads are concatenated, or averaged
/// when `concat == false` (use that for the output layer).
pub struct GatLayer {
    heads: Vec<GatHead>,
    concat: bool,
}

impl GatLayer {
    pub fn new(in_f: i64, out_f_per_head: i64, heads: usize, concat: bool) -> flodl::Result<Self> {
        let mut hs = Vec::new();
        for _ in 0..heads {
            hs.push(GatHead::new(in_f, out_f_per_head)?);
        }
        Ok(GatLayer { heads: hs, concat })
    }

    pub fn out_features(&self, out_f_per_head: i64) -> i64 {
        if self.concat {
            out_f_per_head * self.heads.len() as i64
        } else {
            out_f_per_head
        }
    }

    /// `neg_mask`: additive mask, e.g. [`crate::bridge::attention_mask_variable`].
    pub fn forward_with(&self, x: &Variable, neg_mask: &Variable) -> flodl::Result<Variable> {
        let outs: Vec<Variable> = self
            .heads
            .iter()
            .map(|h| h.forward_with(x, neg_mask))
            .collect::<flodl::Result<_>>()?;
        if self.concat {
            let refs: Vec<&Variable> = outs.iter().collect();
            Variable::cat_many(&refs, outs[0].shape().len() as i32 - 1)
        } else {
            let mut acc = outs[0].clone();
            for o in &outs[1..] {
                acc = acc.add(o)?;
            }
            acc.div_scalar(outs.len() as f64)
        }
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        self.heads.iter().flat_map(|h| h.parameters()).collect()
    }
}

pub struct GatNet {
    hidden: GatLayer,
    out: GatLayer,
    neg_mask: Variable,
    drop: Dropout,
    head: Linear,
    readout: Readout,
}

impl GatNet {
    pub fn new(
        neg_mask: Variable,
        in_f: i64,
        hidden_per_head: i64,
        heads: usize,
        classes: i64,
        dropout: f64,
        readout: Readout,
    ) -> flodl::Result<Self> {
        let hidden = GatLayer::new(in_f, hidden_per_head, heads, true)?;
        let hid_total = hidden.out_features(hidden_per_head);
        Ok(GatNet {
            hidden,
            out: GatLayer::new(hid_total, hidden_per_head, 1, false)?,
            neg_mask,
            drop: Dropout::new(dropout),
            head: Linear::new(hidden_per_head, classes)?,
            readout,
        })
    }
}

impl Module for GatNet {
    fn name(&self) -> &str {
        "gat_net"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let h = self.drop.forward(x)?;
        let h = self
            .drop
            .forward(&self.hidden.forward_with(&h, &self.neg_mask)?.elu(1.0)?)?;
        let mut h = self.out.forward_with(&h, &self.neg_mask)?.elu(1.0)?;
        if self.readout == Readout::GraphMean {
            let node_dim = h.shape().len() as i32 - 2;
            h = h.mean_dim(node_dim, false)?;
        }
        self.head.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.hidden.parameters();
        p.extend(self.out.parameters());
        p.extend(self.head.parameters());
        p
    }

    fn set_training(&self, t: bool) {
        self.drop.set_training(t);
    }
}
