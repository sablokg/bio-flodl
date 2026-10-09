//! Graph construction and normalisation. No libtorch needed: `cargo run --example graph_tour`

use bio_flodl::graph::*;

fn print_matrix(name: &str, m: &[f32], n: usize) {
    println!("{name}:");
    for i in 0..n {
        let row: Vec<String> = (0..n).map(|j| format!("{:6.3}", m[i * n + j])).collect();
        println!("  {}", row.join(" "));
    }
}

fn main() {
    // a small undirected graph: 0-1-2-3 plus a chord 0-2
    let g = Graph::from_edges(4, false, &[(0, 1), (1, 2), (2, 3), (0, 2)]).unwrap();
    print_matrix("adjacency", &g.adjacency(), 4);
    print_matrix("GCN-normalised  D^-1/2 (A+I) D^-1/2", &g.gcn_normalized(), 4);
    print_matrix("row-normalised", &g.row_normalized(), 4);
    print_matrix("Laplacian", &g.laplacian(), 4);
    println!("degrees {:?}", g.degrees());
    println!("edge index {:?}", g.edge_index());
    println!("components {:?}", Graph::from_edges(5, false, &[(0, 1), (3, 4)]).unwrap().connected_components());

    // residue contact map from 3-D coordinates (an ideal alpha-helix-like trace, ~3.8 A spacing)
    let coords: Vec<[f32; 3]> = (0..12)
        .map(|i| {
            let t = i as f32 * 100f32.to_radians();
            [2.3 * t.cos(), 2.3 * t.sin(), 1.5 * i as f32]
        })
        .collect();
    let dist = distance_matrix(&coords);
    let contacts = from_contact_map(&dist, coords.len(), 8.0, 3).unwrap();
    println!("contact graph: {} residues, {} contacts (cutoff 8 A, |i-j| >= 3)", contacts.n, contacts.edges.len());
    println!("chain graph edges: {}", chain(12).edges.len());

    // de Bruijn graph from reads
    let reads: [&[u8]; 2] = [b"ACGTACGT", b"CGTACGTT"];
    let (nodes, db) = de_bruijn(&reads, 4).unwrap();
    println!("de Bruijn (k=4): {} nodes, {} edges", nodes.len(), db.edges.len());
    for (u, v, w) in &db.edges {
        println!("  {} -> {}  (x{})", String::from_utf8_lossy(&nodes[*u]), String::from_utf8_lossy(&nodes[*v]), w);
    }
}
