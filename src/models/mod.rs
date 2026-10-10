//! Model zoo built on flodl layers. Every model implements `flodl::Module`, so it works
//! with flodl optimizers, checkpoints and [`crate::train`].
//!
//! Input layouts:
//! * CNNs: `[batch, alphabet, length]` (channels-first)
//! * RNNs / Transformer / MLP: `[batch, length, alphabet]`

pub mod cnn;
pub mod compose;
pub mod mlp;
pub mod rnn;
pub mod transformer;

/*
Gaurav Sablok
gsablok@proton.me
*/

use flodl::{Device, Tensor, Variable};
use std::cell::RefCell;

pub(crate) fn from_f32(data: &[f32], shape: &[i64]) -> flodl::Result<Tensor> {
    Tensor::from_f32(data, shape, Device::CPU)
}

/// A fixed input tensor held by a model, such as the transformer's positional table.
/// It is neither a parameter nor a buffer, so it is not trained and not written to
/// checkpoints. It follows the device of the input: the first forward after a device change
/// moves it once, later calls reuse the moved copy.
pub(crate) struct Constant(RefCell<Variable>);

impl Constant {
    pub(crate) fn new(v: Variable) -> Self {
        Constant(RefCell::new(v))
    }

    pub(crate) fn on(&self, device: Device) -> flodl::Result<Variable> {
        if self.0.borrow().device() != device {
            let moved = self.0.borrow().to_device(device)?;
            *self.0.borrow_mut() = moved;
        }
        Ok(self.0.borrow().clone())
    }
}
