//! Signature-only stub of flodl 0.8 (from docs.rs) used to type-check bio-flodl without libtorch.
#![allow(unused_variables, clippy::all)]
use std::rc::Rc;

#[derive(Debug)] pub struct TensorError;
pub type Result<T> = std::result::Result<T, TensorError>;
#[derive(Clone, Copy, Debug, PartialEq)] pub enum Device { CPU, CUDA(u8) }
#[derive(Default, Clone)] pub struct TensorOptions;
#[derive(Clone)] pub struct Tensor;
impl Tensor {
    pub fn randn(shape: &[i64], opts: TensorOptions) -> Result<Tensor> { todo!() }
    pub fn from_f32(data: &[f32], shape: &[i64], device: Device) -> Result<Tensor> { todo!() }
    pub fn from_i64(data: &[i64], shape: &[i64], device: Device) -> Result<Tensor> { todo!() }
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> { todo!() }
}

#[derive(Clone)] pub struct Variable;
macro_rules! v0 { ($($n:ident),*) => { $(pub fn $n(&self) -> Result<Variable> { todo!() })* } }
macro_rules! vs { ($($n:ident),*) => { $(pub fn $n(&self, s: f64) -> Result<Variable> { todo!() })* } }
macro_rules! vv { ($($n:ident),*) => { $(pub fn $n(&self, o: &Variable) -> Result<Variable> { todo!() })* } }
macro_rules! vd { ($($n:ident),*) => { $(pub fn $n(&self, dim: i32, keepdim: bool) -> Result<Variable> { todo!() })* } }
impl Variable {
    pub fn new(data: Tensor, requires_grad: bool) -> Self { todo!() }
    v0!(relu, sigmoid, tanh, gelu, neg, sum, mean, sign, exp, log, sqrt);
    vs!(mul_scalar, div_scalar, add_scalar, leaky_relu, elu);
    vv!(add, sub, mul, div, matmul);
    vd!(sum_dim, mean_dim, max_dim, min_dim);
    pub fn softmax(&self, dim: i32) -> Result<Variable> { todo!() }
    pub fn log_softmax(&self, dim: i32) -> Result<Variable> { todo!() }
    pub fn reshape(&self, shape: &[i64]) -> Result<Variable> { todo!() }
    pub fn transpose(&self, d0: i32, d1: i32) -> Result<Variable> { todo!() }
    pub fn narrow(&self, dim: i32, start: i64, length: i64) -> Result<Variable> { todo!() }
    pub fn index_select(&self, dim: i32, index: &Tensor) -> Result<Variable> { todo!() }
    pub fn gather(&self, dim: i32, index: &Tensor) -> Result<Variable> { todo!() }
    pub fn cat_many(vars: &[&Variable], dim: i32) -> Result<Variable> { todo!() }
    pub fn data(&self) -> Tensor { todo!() }
    pub fn shape(&self) -> Vec<i64> { todo!() }
    pub fn device(&self) -> Device { todo!() }
    pub fn item(&self) -> Result<f64> { todo!() }
    pub fn backward(&self) -> Result<()> { todo!() }
}

#[derive(Clone)] pub struct Parameter { pub variable: Variable, pub name: String }
impl Parameter { pub fn new(data: Tensor, name: &str) -> Self { todo!() } }

pub trait Module {
    fn forward(&self, input: &Variable) -> Result<Variable>;
    fn parameters(&self) -> Vec<Parameter> { Vec::new() }
    fn name(&self) -> &str { "module" }
    fn sub_modules(&self) -> Vec<Rc<dyn Module>> { Vec::new() }
    fn set_training(&self, _training: bool) {}
    fn train(&self) { self.set_training(true) }
    fn eval(&self) { self.set_training(false) }
}

macro_rules! leaf { ($t:ident) => { pub struct $t; impl Module for $t { fn forward(&self, i: &Variable) -> Result<Variable> { todo!() } fn parameters(&self) -> Vec<Parameter> { todo!() } } } }
leaf!(Linear);
pub struct Conv1d { pub weight: Parameter, pub bias: Option<Parameter> }
impl Module for Conv1d { fn forward(&self, i: &Variable) -> Result<Variable> { todo!() } fn parameters(&self) -> Vec<Parameter> { todo!() } }
 leaf!(LayerNorm); leaf!(MultiheadAttention); leaf!(LSTM); leaf!(GRU); leaf!(Dropout);
impl Linear { pub fn new(i: i64, o: i64) -> Result<Self> { todo!() } }
impl Conv1d {
    pub fn new(i: i64, o: i64, k: i64) -> Result<Self> { todo!() }
    #[allow(clippy::too_many_arguments)]
    pub fn build(i: i64, o: i64, k: i64, with_bias: bool, stride: i64, padding: i64, dilation: i64, groups: i64, device: Device) -> Result<Self> { todo!() }
}
impl Conv1d { }
impl LayerNorm { pub fn new(size: i64) -> Result<Self> { todo!() } }
impl MultiheadAttention {
    pub fn new(e: i64, h: i64) -> Result<Self> { todo!() }
    pub fn forward_ext(&self, q: &Variable, k: &Variable, v: &Variable, mask: Option<&Tensor>) -> Result<Variable> { todo!() }
}
impl LSTM { pub fn new(i: i64, h: i64, layers: usize) -> Result<Self> { todo!() } pub fn batch_first(self, b: bool) -> Self { self } }
impl GRU { pub fn new(i: i64, h: i64, layers: usize) -> Result<Self> { todo!() } pub fn batch_first(self, b: bool) -> Self { self } }
impl Dropout { pub fn new(p: f64) -> Self { todo!() } }

pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&self);
    fn lr(&self) -> f64;
    fn set_lr(&mut self, lr: f64);
    fn scale_lr(&mut self, factor: f64) {}
}
pub struct Adam;
impl Adam { pub fn new(params: &[Parameter], lr: f64) -> Self { todo!() } }
impl Optimizer for Adam { fn step(&mut self) -> Result<()> { todo!() } fn zero_grad(&self) {} fn lr(&self) -> f64 { 0.0 } fn set_lr(&mut self, lr: f64) {} }

pub fn cross_entropy_loss(pred: &Variable, target: &Variable) -> Result<Variable> { todo!() }
pub fn clip_grad_norm(params: &[Parameter], max_norm: f64) -> Result<f64> { todo!() }
