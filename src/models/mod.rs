//! Model zoo built on flodl layers. Every model implements `flodl::Module`, so it works
//! with flodl optimizers, checkpoints and [`crate::train`].
//!
//! Input layouts:
//! * CNNs: `[batch, alphabet, length]` (channels-first)
//! * RNNs / Transformer / MLP: `[batch, length, alphabet]`
//! * GNNs: node features `[nodes, features]` or `[batch, nodes, features]`
//!   (a batch shares one adjacency matrix)

pub mod cnn;
pub mod compose;
pub mod gnn;
pub mod mlp;
pub mod rnn;
pub mod transformer;

/*
Gaurav Sablok
gsablok@proton.me
*/

use flodl::{Device, Parameter, Tensor, Variable};

pub(crate) fn glorot(
    shape: &[i64],
    fan_in: i64,
    fan_out: i64,
    name: &str,
) -> flodl::Result<Parameter> {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let w = Variable::new(Tensor::randn(shape, Default::default())?, false).mul_scalar(std)?;
    Ok(Parameter::new(w.data(), name))
}

pub(crate) fn zeros_param(shape: &[i64], name: &str) -> flodl::Result<Parameter> {
    let z = Variable::new(Tensor::randn(shape, Default::default())?, false).mul_scalar(0.0)?;
    Ok(Parameter::new(z.data(), name))
}

pub(crate) fn from_f32(data: &[f32], shape: &[i64]) -> flodl::Result<Tensor> {
    Tensor::from_f32(data, shape, Device::CPU)
}
