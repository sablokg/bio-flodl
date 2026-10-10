//! Conversions from the pure-Rust core into flodl tensors (feature `flodl`).
//! All tensors are created on the CPU; move models/inputs with `move_to_device` / `to_device`.

use crate::encode::{Batch, OneHot};
use flodl::{Device, Tensor, Variable};

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn tensor_f32(data: &[f32], shape: &[i64]) -> flodl::Result<Tensor> {
    Tensor::from_f32(data, shape, Device::CPU)
}

pub fn variable_f32(data: &[f32], shape: &[i64]) -> flodl::Result<Variable> {
    Ok(Variable::new(tensor_f32(data, shape)?, false))
}

/// Int64 class-index vector `[n]`, as accepted by `cross_entropy_loss`.
pub fn labels_variable(y: &[i64]) -> flodl::Result<Variable> {
    Ok(Variable::new(
        Tensor::from_i64(y, &[y.len() as i64], Device::CPU)?,
        false,
    ))
}

pub fn onehot_to_variable(oh: &OneHot) -> flodl::Result<Variable> {
    variable_f32(&oh.data, &[oh.len as i64, oh.width as i64])
}

/// `[n, L, width]` (RNN / Transformer layout).
pub fn batch_to_variable(b: &Batch) -> flodl::Result<Variable> {
    variable_f32(&b.data, &[b.n as i64, b.len as i64, b.width as i64])
}

/// `[n, width, L]` (Conv1d layout).
pub fn batch_to_variable_channels_first(b: &Batch) -> flodl::Result<Variable> {
    variable_f32(
        &b.channels_first(),
        &[b.n as i64, b.width as i64, b.len as i64],
    )
}
