//! Graph classification with multi-head graph attention on the same contact graph.
//! `cargo run --release --features flodl --example gat_graph`

mod common;
use flodl_bio::bridge::attention_mask_variable;
use flodl_bio::data::synthetic_graph_task;
use flodl_bio::graph::{distance_matrix, from_contact_map};
use flodl_bio::models::gnn::{GatNet, Readout};
use flodl_bio::train::{adam, fit, report, TrainConfig};

fn main() -> flodl::Result<()> {
    let nodes = 24;
    let coords: Vec<[f32; 3]> = (0..nodes)
        .map(|i| { let t = i as f32 * 100f32.to_radians(); [2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f32] })
        .collect();
    let graph = from_contact_map(&distance_matrix(&coords), nodes, 8.0, 2).unwrap();

    let (train, val) = synthetic_graph_task(1200, nodes, 4, 5).split(0.2, 1);
    let model = GatNet::new(attention_mask_variable(&graph)?, 4, 8, 4, 2, 0.1, Readout::GraphMean)?;
    let mut opt = adam(&model, 5e-3);
    fit(&model, &mut opt, &train, Some(&val), &TrainConfig { epochs: 20, batch_size: 64, patience: Some(5), ..Default::default() })?;
    common::print_report("GatNet", &report(&model, &val, 2, 64)?);
    Ok(())
}
