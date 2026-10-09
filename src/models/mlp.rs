//! Fully connected baseline over a flattened one-hot window.

use flodl::{Dropout, Linear, Module, Parameter, Variable};

/*
Gaurav Sablok
gsablok@proton.me
*/

pub struct FlatMlp {
    layers: Vec<Linear>,
    drop: Dropout,
}

impl FlatMlp {
    /// `input_features = length * alphabet`.
    pub fn new(
        input_features: i64,
        hidden: &[i64],
        classes: i64,
        dropout: f64,
    ) -> flodl::Result<Self> {
        let mut dims = vec![input_features];
        dims.extend_from_slice(hidden);
        dims.push(classes);
        let mut layers = Vec::new();
        for w in dims.windows(2) {
            layers.push(Linear::new(w[0], w[1])?);
        }
        Ok(FlatMlp {
            layers,
            drop: Dropout::new(dropout),
        })
    }
}

impl Module for FlatMlp {
    fn name(&self) -> &str {
        "flat_mlp"
    }

    fn forward(&self, x: &Variable) -> flodl::Result<Variable> {
        let b = x.shape()[0];
        let mut h = x.reshape(&[b, -1])?;
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(&h)?;
            if i != last {
                h = self.drop.forward(&h.gelu()?)?;
            }
        }
        Ok(h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
    fn set_training(&self, t: bool) {
        self.drop.set_training(t);
    }
}
