//! Graph classification with a GCN over a residue contact graph shared by all samples.
//! The graph comes from synthetic helix-like coordinates (see `graph::from_contact_map`).
//! `cargo run --release --features flodl --example gcn_graph`

mod common;
use flodl_bio::bridge::gcn_adjacency_variable;
use flodl_bio::data::synthetic_graph_task;
use flodl_bio::graph::{distance_matrix, from_contact_map};
use flodl_bio::models::gnn::{GcnNet, Readout};
use flodl_bio::train::{adam, fit, report, TrainConfig};

fn main() -> flodl::Result<()> {
    let nodes = 24;
    let coords: Vec<[f32; 3]> = (0..nodes)
        .map(|i| { let t = i as f32 * 100f32.to_radians(); [2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f32] })
        .collect();
    let graph = from_contact_map(&distance_matrix(&coords), nodes, 8.0, 2).unwrap();
    println!("contact graph: {} nodes, {} edges", graph.n, graph.edges.len());

    let (train, val) = synthetic_graph_task(1200, nodes, 4, 5).split(0.2, 1);
    let model = GcnNet::new(gcn_adjacency_variable(&graph)?, 4, 32, 2, 2, 0.1, Readout::GraphMean)?;
    let mut opt = adam(&model, 1e-2);
    fit(&model, &mut opt, &train, Some(&val), &TrainConfig { epochs: 20, batch_size: 64, patience: Some(5), ..Default::default() })?;
    common::print_report("GcnNet", &report(&model, &val, 2, 64)?);
    Ok(())
}
