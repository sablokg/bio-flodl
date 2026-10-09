//! Training and inference throughput of every bundled model (CPU tensors).
//!   cargo run --release --features flodl --example bench_models -- [--batch 64] [--len 200] [--steps 20] [--nodes 64]
//!
//! Reports ms per training step (forward + loss + backward + Adam step), ms per forward pass in eval
//! mode, samples/s, and parameter count. Compare against a PyTorch reference of the same architecture
//! and batch size on the same machine for the "throughput" row of the evaluation.

use bio_flodl::bridge::{attention_mask_variable, gcn_adjacency_variable, labels_variable};
use bio_flodl::data::Rng;
use bio_flodl::graph::Graph;
use bio_flodl::models::cnn::*;
use bio_flodl::models::gnn::*;
use bio_flodl::models::mlp::FlatMlp;
use bio_flodl::models::rnn::{BiRnnClassifier, RnnKind};
use bio_flodl::models::transformer::{TransformerClassifier, TransformerConfig};
use bio_flodl::train::adam;
use flodl::{cross_entropy_loss, Module, Optimizer, Tensor, Variable};
use std::time::Instant;

fn arg(name: &str, default: usize) -> usize {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn randn(shape: &[i64]) -> flodl::Result<Variable> { Ok(Variable::new(Tensor::randn(shape, Default::default())?, false)) }

struct Row { name: String, params: i64, train_ms: f64, infer_ms: f64, batch: usize }

fn bench<M: Module>(name: &str, model: &M, x: &Variable, y: &Variable, batch: usize, steps: usize) -> flodl::Result<Row> {
    let params: i64 = model.parameters().iter().map(|p| p.variable.shape().iter().product::<i64>()).sum();
    let mut opt = adam(model, 1e-3);

    model.train();
    let mut step = || -> flodl::Result<()> {
        let loss = cross_entropy_loss(&model.forward(x)?, y)?;
        opt.zero_grad();
        loss.backward()?;
        opt.step()?;
        loss.item()?; // forces completion
        Ok(())
    };
    for _ in 0..3 { step()?; }
    let t = Instant::now();
    for _ in 0..steps { step()?; }
    let train_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;

    model.eval();
    for _ in 0..3 { model.forward(x)?.sum()?.item()?; }
    let t = Instant::now();
    for _ in 0..steps { model.forward(x)?.sum()?.item()?; }
    let infer_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;

    Ok(Row { name: name.into(), params, train_ms, infer_ms, batch })
}

fn main() -> flodl::Result<()> {
    let (b, l, steps, nodes) = (arg("--batch", 64), arg("--len", 200), arg("--steps", 20), arg("--nodes", 64));
    let (bi, li, ni) = (b as i64, l as i64, nodes as i64);
    let mut rng = Rng::new(1);
    let labels: Vec<i64> = (0..b).map(|_| rng.below(2) as i64).collect();
    let y = labels_variable(&labels)?;
    println!("batch {b}, sequence length {l}, graph nodes {nodes}, {steps} timed steps, CPU\n");

    let x_cf = randn(&[bi, 4, li])?; // [B, 4, L]
    let x_cl = randn(&[bi, li, 4])?; // [B, L, 4]
    let mut rows = Vec::new();

    rows.push(bench("MotifCnn (3 widths x 64)", &MotifCnn::new(&MotifCnnConfig { in_channels: 4, filters: 64, kernel_sizes: vec![8, 12, 16], dropout: 0.3, classes: 2 })?, &x_cf, &y, b, steps)?);
    rows.push(bench("DilatedResCnn (64 ch, 6 blocks)", &DilatedResCnn::new(&DilatedCnnConfig { in_channels: 4, channels: 64, kernel: 3, blocks: 6, dropout: 0.3, classes: 2 })?, &x_cf, &y, b, steps)?);
    rows.push(bench("BiLSTM (hidden 64)", &BiRnnClassifier::new(RnnKind::Lstm, 4, 64, 1, 0.3, 2)?, &x_cl, &y, b, steps)?);
    rows.push(bench("BiGRU (hidden 64)", &BiRnnClassifier::new(RnnKind::Gru, 4, 64, 1, 0.3, 2)?, &x_cl, &y, b, steps)?);
    rows.push(bench("Transformer (d64, 4 heads, 2 layers)", &TransformerClassifier::new(&TransformerConfig { alphabet: 4, d_model: 64, heads: 4, layers: 2, d_ff: 128, max_len: l, dropout: 0.1, classes: 2 })?, &x_cl, &y, b, steps)?);
    rows.push(bench("FlatMlp (2 x 256)", &FlatMlp::new(4 * li, &[256, 256], 2, 0.3)?, &x_cl, &y, b, steps)?);

    // graphs: a ring plus random chords, features [B, N, 16]
    let mut g = Graph::new(nodes, false);
    for i in 0..nodes { g.add_edge(i, (i + 1) % nodes, 1.0).unwrap(); g.add_edge(i, rng.below(nodes), 1.0).unwrap(); }
    let x_g = randn(&[bi, ni, 16])?;
    rows.push(bench("GcnNet (2 layers, 64 hidden)", &GcnNet::new(gcn_adjacency_variable(&g)?, 16, 64, 2, 2, 0.1, Readout::GraphMean)?, &x_g, &y, b, steps)?);
    rows.push(bench("GatNet (4 heads x 16)", &GatNet::new(attention_mask_variable(&g)?, 16, 16, 4, 2, 0.1, Readout::GraphMean)?, &x_g, &y, b, steps)?);

    println!("| model | parameters | train ms/step | train samples/s | infer ms/batch | infer samples/s |");
    println!("|---|---|---|---|---|---|");
    for r in &rows {
        println!("| {} | {} | {:.2} | {:.0} | {:.2} | {:.0} |", r.name, r.params, r.train_ms,
            r.batch as f64 / (r.train_ms / 1e3), r.infer_ms, r.batch as f64 / (r.infer_ms / 1e3));
    }
    Ok(())
}
