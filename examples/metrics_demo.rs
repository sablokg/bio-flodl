//! Datasets, splitting and metrics. No libtorch needed: `cargo run --example metrics_demo`

use flodl_bio::alphabet::Alphabet;
use flodl_bio::data::*;

fn main() {
    let (seqs, labels) = synthetic_motif(1000, 100, b"TATAAA", 7);
    let refs: Vec<&[u8]> = seqs.iter().map(|s| s.as_slice()).collect();
    let ds = Dataset::from_sequences(&refs, &labels, &Alphabet::DNA, 100, true).unwrap();
    let (train, val) = ds.split(0.2, 1);
    println!("train {} / val {}, sample shape {:?}", train.n, val.n, train.sample_shape);

    // a trivial "classifier": does the sequence contain the motif? (perfect on this task)
    let scores: Vec<f32> = val.x.chunks(val.sample_len()).map(|_| 0.5).collect();
    println!("constant scores -> AUROC {:?}, AUPRC {:?}", auroc(&scores, &val.y), average_precision(&scores, &val.y));

    // a noisy scorer built from labels + a seeded RNG
    let mut rng = Rng::new(3);
    let noisy: Vec<f32> = val.y.iter().map(|&y| y as f32 * 0.4 + (rng.below(1000) as f32) / 1000.0).collect();
    let pred: Vec<usize> = noisy.iter().map(|&s| (s > 0.7) as usize).collect();
    let cm = confusion_matrix(&pred, &val.y, 2);
    println!("noisy scorer: acc {:.3}  macro-F1 {:.3}  MCC {:.3}", accuracy(&pred, &val.y), macro_f1(&cm, 2), mcc_binary(&cm));
    println!("              AUROC {:.3}  AUPRC {:.3}", auroc(&noisy, &val.y).unwrap(), average_precision(&noisy, &val.y).unwrap());
    println!("confusion matrix (rows = true) {:?}", cm);
}
