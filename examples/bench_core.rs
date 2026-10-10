//! Throughput of the pure-Rust data layer. No libtorch needed.
//! Run in release mode: `cargo run --release --example bench_core [-- --scale 1.0]`

use flodl_bio::alphabet::Alphabet;
use flodl_bio::data::*;
use flodl_bio::encode::*;
use flodl_bio::graph::*;
use std::hint::black_box;
use std::time::Instant;

fn random_dna(rng: &mut Rng, n: usize) -> Vec<u8> { (0..n).map(|_| b"ACGT"[rng.below(4)]).collect() }

/// Best of `reps` runs, in seconds.
fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    (0..reps).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min)
}

fn row(name: &str, work: f64, unit: &str, secs: f64) {
    println!("| {name:<44} | {:>10.3} ms | {:>12.2} M {unit}/s |", secs * 1e3, work / secs / 1e6);
}

fn main() {
    let scale: f64 = std::env::args().skip_while(|a| a != "--scale").nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let sc = |n: usize| ((n as f64) * scale).max(1.0) as usize;
    let mut rng = Rng::new(1);
    let reps = 5;

    println!("flodl-bio core benchmark (best of {reps}, scale {scale})\n");
    println!("| {:<44} | {:>13} | {:>22} |", "operation", "time", "throughput");
    println!("|{:-<46}|{:-<15}|{:-<24}|", "", "", "");

    // --- encoding
    let seq_len = 1000;
    let n_seq = sc(2000);
    let seqs: Vec<Vec<u8>> = (0..n_seq).map(|_| random_dna(&mut rng, seq_len)).collect();
    let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
    let bases = (n_seq * seq_len) as f64;

    row(&format!("one_hot ({n_seq} x {seq_len} bp, one by one)"), bases, "bases", time(reps, || {
        for s in &refs { black_box(one_hot(s, &Alphabet::DNA).unwrap()); }
    }));
    row(&format!("one_hot_batch + channels_first ({n_seq} x {seq_len})"), bases, "bases", time(reps, || {
        let b = one_hot_batch(&refs, &Alphabet::DNA, None);
        black_box(b.channels_first());
    }));
    let genome = random_dna(&mut rng, sc(2_000_000));
    row(&format!("kmer_counts k=6 ({} bp)", genome.len()), genome.len() as f64, "bases", time(reps, || {
        black_box(kmer_counts(&genome, 6, &Alphabet::DNA).unwrap());
    }));
    row(&format!("reverse_complement ({} bp)", genome.len()), genome.len() as f64, "bases", time(reps, || {
        black_box(flodl_bio::alphabet::reverse_complement(&genome));
    }));

    // --- graphs
    let n = sc(500).min(1500);
    let mut g = Graph::new(n, false);
    for i in 0..n { for _ in 0..5 { let j = rng.below(n); if i != j { g.add_edge(i, j, 1.0).unwrap(); } } }
    let cells = (n * n) as f64;
    row(&format!("adjacency (n={n}, {} edges)", g.edges.len()), cells, "cells", time(reps, || { black_box(g.adjacency()); }));
    row(&format!("gcn_normalized (n={n})"), cells, "cells", time(reps, || { black_box(g.gcn_normalized()); }));
    row(&format!("laplacian (n={n})"), cells, "cells", time(reps, || { black_box(g.laplacian()); }));
    let coords: Vec<[f32; 3]> = (0..n).map(|_| [rng.below(1000) as f32 / 10.0, rng.below(1000) as f32 / 10.0, rng.below(1000) as f32 / 10.0]).collect();
    row(&format!("distance_matrix + contact graph (n={n})"), cells, "cells", time(reps, || {
        let d = distance_matrix(&coords);
        black_box(from_contact_map(&d, n, 8.0, 3).unwrap());
    }));
    let reads: Vec<&[u8]> = refs.iter().take(sc(200)).copied().collect();
    let read_bases = (reads.len() * seq_len) as f64;
    row(&format!("de_bruijn k=21 ({} reads x {seq_len})", reads.len()), read_bases, "bases", time(reps, || {
        black_box(de_bruijn(&reads, 21).unwrap());
    }));

    // --- data / metrics
    let m = sc(1_000_000);
    let scores: Vec<f32> = (0..m).map(|_| rng.below(1_000_000) as f32 / 1e6).collect();
    let labels: Vec<i64> = (0..m).map(|_| (rng.below(2)) as i64).collect();
    row(&format!("auroc ({m} scores)"), m as f64, "scores", time(reps, || { black_box(auroc(&scores, &labels)); }));
    row(&format!("average_precision ({m} scores)"), m as f64, "scores", time(reps, || { black_box(average_precision(&scores, &labels)); }));
    let ds = Dataset::from_sequences(&refs, &vec![0; n_seq], &Alphabet::DNA, seq_len, true).unwrap();
    row(&format!("Dataset::split + batch gather ({n_seq} x {seq_len})"), bases, "bases", time(reps, || {
        let (tr, _) = ds.split(0.2, 1);
        for idx in batch_indices(tr.n, 64, None) { black_box(tr.batch(&idx)); }
    }));
}
