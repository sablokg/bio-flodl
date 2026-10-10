use bio_flodl::{alphabet::*, encode::*, graph::*};

#[test]
fn one_hot_dna() {
    let oh = one_hot(b"acgt", &Alphabet::DNA).unwrap();
    assert_eq!(oh.shape(), [4, 4]);
    assert_eq!(&oh.data[0..4], &[1., 0., 0., 0.]);
    assert_eq!(&oh.data[12..16], &[0., 0., 0., 1.]);
}

#[test]
fn unknown_symbol_errors_and_lenient_zeros() {
    assert!(one_hot(b"ACN", &Alphabet::DNA).is_err());
    let oh = one_hot_lenient(b"ACN", &Alphabet::DNA);
    assert_eq!(&oh.data[8..12], &[0., 0., 0., 0.]);
}

#[test]
fn channels_first_roundtrip() {
    let oh = one_hot(b"AC", &Alphabet::DNA).unwrap();
    // channels: A=[1,0], C=[0,1], G=[0,0], T=[0,0]
    assert_eq!(oh.channels_first(), vec![1., 0., 0., 1., 0., 0., 0., 0.]);
}

#[test]
fn batch_padding_and_mask() {
    let b = one_hot_batch(&[b"AC", b"A"], &Alphabet::DNA, None);
    assert_eq!(b.shape(), [2, 2, 4]);
    assert_eq!(b.mask, vec![1., 1., 1., 0.]);
}

#[test]
fn kmers() {
    let v = kmer_counts(b"AAAC", 2, &Alphabet::DNA).unwrap();
    assert_eq!(v.len(), 16);
    assert_eq!(v[0], 2.0); // AA
    assert_eq!(v[1], 1.0); // AC
}

#[test]
fn revcomp_gc() {
    assert_eq!(reverse_complement(b"AACG"), b"CGTT");
    assert!((gc_content(b"GCAT") - 0.5).abs() < 1e-6);
}

#[test]
fn adjacency_and_gcn() {
    let g = Graph::from_edges(3, false, &[(0, 1), (1, 2)]).unwrap();
    assert_eq!(g.adjacency(), vec![0., 1., 0., 1., 0., 1., 0., 1., 0.]);
    assert_eq!(g.degrees(), vec![1., 2., 1.]);
    let n = g.gcn_normalized();
    assert!((n[0] - 0.5).abs() < 1e-6); // 1/sqrt(2)*1/sqrt(2)
    let l = g.laplacian();
    assert_eq!(l[0], 1.0);
    assert_eq!(l[1], -1.0);
}

#[test]
fn dense_roundtrip_and_components() {
    let g = Graph::from_edges(4, false, &[(0, 1), (2, 3)]).unwrap();
    let g2 = Graph::from_dense(&g.adjacency(), 4, false).unwrap();
    assert_eq!(g2.adjacency(), g.adjacency());
    let c = g.connected_components();
    assert_eq!(c[0], c[1]);
    assert_ne!(c[1], c[2]);
}

#[test]
fn contact_map() {
    let coords = [[0., 0., 0.], [3.8, 0., 0.], [7.6, 0., 0.], [0., 5., 0.]];
    let d = distance_matrix(&coords);
    let g = from_contact_map(&d, 4, 8.0, 2).unwrap();
    assert!(g.edges.iter().all(|&(i, j, _)| j - i >= 2));
}

#[test]
fn de_bruijn_graph() {
    let (labels, g) = de_bruijn(&[b"ACGTA"], 3).unwrap();
    assert_eq!(labels.len(), 4); // AC CG GT TA
    assert_eq!(g.edges.len(), 3);
}

use bio_flodl::data::*;

#[test]
fn rng_is_reproducible_and_shuffles() {
    let mut a = Rng::new(7);
    let mut b = Rng::new(7);
    assert_eq!(a.next_u64(), b.next_u64());
    let mut v: Vec<usize> = (0..50).collect();
    Rng::new(1).shuffle(&mut v);
    let mut s = v.clone();
    s.sort();
    assert_eq!(s, (0..50).collect::<Vec<_>>());
    assert_ne!(v, s);
}

#[test]
fn dataset_shapes_split_and_batches() {
    let seqs: Vec<&[u8]> = vec![b"ACGT", b"TTTT", b"GGCC", b"AAAA"];
    let ds = Dataset::from_sequences(&seqs, &[0, 1, 0, 1], &Alphabet::DNA, 4, true).unwrap();
    assert_eq!(ds.sample_shape, vec![4, 4]);
    let (tr, va) = ds.split(0.25, 3);
    assert_eq!((tr.n, va.n), (3, 1));
    let b = batch_indices(10, 4, None);
    assert_eq!(b.len(), 3);
    assert_eq!(b[2].len(), 2);
    assert!(Dataset::new(vec![0.0; 5], vec![2], vec![0, 1]).is_err());
}

#[test]
fn synthetic_motif_planted() {
    let (s, y) = synthetic_motif(20, 30, b"TATAAA", 5);
    for (seq, label) in s.iter().zip(&y) {
        let has = seq.windows(6).any(|w| w == b"TATAAA");
        if *label == 1 { assert!(has); }
    }
    assert_eq!(y.iter().sum::<i64>(), 10);
}

#[test]
fn metrics_values() {
    assert_eq!(accuracy(&[0, 1, 1], &[0, 1, 0]), 2.0 / 3.0);
    let cm = confusion_matrix(&[0, 1, 1, 0], &[0, 1, 0, 0], 2);
    assert_eq!(cm, vec![2, 1, 0, 1]);
    assert!(macro_f1(&cm, 2) > 0.7);
    assert!((mcc_binary(&[5, 0, 0, 5]) - 1.0).abs() < 1e-9);
    assert_eq!(auroc(&[0.1, 0.4, 0.35, 0.8], &[0, 0, 1, 1]), Some(0.75));
    assert_eq!(auroc(&[0.5, 0.5], &[0, 1]), Some(0.5));
    assert_eq!(auroc(&[0.5, 0.6], &[1, 1]), None);
}

#[test]
fn positional_encoding_and_attn_mask() {
    let pe = bio_flodl::encode::sinusoidal_positions(4, 4);
    assert_eq!(pe.len(), 16);
    assert_eq!(&pe[0..4], &[0.0, 1.0, 0.0, 1.0]); // position 0: sin 0, cos 0
    let g = Graph::from_edges(3, false, &[(0, 1)]).unwrap();
    let m = g.attention_mask();
    assert_eq!(m[0], 0.0);      // self loop
    assert_eq!(m[1], 0.0);      // edge 0-1
    assert_eq!(m[2], -1e9);     // no edge 0-2
}

#[test]
fn average_precision_and_fasta_and_graph_task() {
    let ap = average_precision(&[0.1, 0.4, 0.35, 0.8], &[0, 0, 1, 1]).unwrap();
    assert!((ap - 0.8333333).abs() < 1e-6);
    assert_eq!(average_precision(&[0.9, 0.8], &[1, 1]), None);
    assert!((average_precision(&[0.5, 0.5], &[0, 1]).unwrap() - 0.5).abs() < 1e-9);

    let recs = bio_flodl::io::parse_fasta(">a desc\nacg\nT\n\n>b\nGG\n");
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].1, b"ACGT");
    assert_eq!(bio_flodl::io::fit_length(b"ACGT", 6), b"ACGTNN");
    assert_eq!(bio_flodl::io::fit_length(b"AACGTT", 2), b"CG");

    let ds = synthetic_graph_task(10, 9, 4, 3);
    assert_eq!(ds.sample_shape, vec![9, 4]);
    assert_eq!(ds.x.len(), 10 * 9 * 4);
    // exactly one hot per node
    assert!(ds.x.chunks(4).all(|c| c.iter().sum::<f32>() == 1.0));
}

#[test]
fn curves_match_their_summary_metrics() {
    // Ties with mixed labels (0.4) and same labels (0.35) exercise the grouping.
    let scores = [0.1, 0.4, 0.35, 0.8, 0.4, 0.9, 0.2, 0.35];
    let labels = [0, 0, 1, 1, 1, 0, 0, 1];

    let roc = roc_curve(&scores, &labels).unwrap();
    assert_eq!(roc.first(), Some(&(0.0, 0.0)));
    assert_eq!(roc.last(), Some(&(1.0, 1.0)));
    let area: f64 = roc
        .windows(2)
        .map(|w| (w[1].0 - w[0].0) * (w[1].1 + w[0].1) / 2.0)
        .sum();
    assert!((area - auroc(&scores, &labels).unwrap()).abs() < 1e-12);

    let pr = pr_curve(&scores, &labels).unwrap();
    assert_eq!(pr.last().unwrap().0, 1.0);
    let (mut prev, mut steps) = (0.0, 0.0);
    for &(recall, precision) in &pr {
        steps += (recall - prev) * precision;
        prev = recall;
    }
    assert!((steps - average_precision(&scores, &labels).unwrap()).abs() < 1e-12);

    assert!(roc_curve(&[0.5, 0.6], &[1, 1]).is_none());
    assert!(pr_curve(&[0.5, 0.6], &[0, 0]).is_none());
}

#[test]
fn motif_pfms_background_and_meme() {
    use bio_flodl::motif::{background, pfms_from_activations, write_meme};
    let seqs: [&[u8]; 3] = [b"ACGTTT", b"TTACGG", b"GGGTAC"];
    let x = one_hot_batch(&seqs, &Alphabet::DNA, None).channels_first();
    let (n, alphabet, len, width, filters) = (3, 4, 6, 2, 2);
    let positions = len - width + 1;
    // Filter 0 fires on the three "AC" windows and weakly elsewhere; filter 1 never fires.
    let mut acts = vec![0.0f32; n * filters * positions];
    for (s, hit) in [(0, 0), (1, 2), (2, 4)] {
        for p in 0..positions {
            acts[s * filters * positions + p] = if p == hit { 1.0 } else { 0.1 };
        }
    }
    let pfms = pfms_from_activations(&x, n, alphabet, len, &acts, filters, width, 0.5);
    assert_eq!(pfms[0].sites, 3);
    assert_eq!(pfms[0].probs, vec![1., 0., 0., 0., 0., 1., 0., 0.]);
    assert_eq!(pfms[0].consensus(b"ACGT"), "AC");
    assert_eq!(pfms[1].sites, 0);

    let bg = background(&x, n, alphabet, len);
    let want = [3.0 / 18.0, 3.0 / 18.0, 6.0 / 18.0, 6.0 / 18.0];
    assert!(bg.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12));

    let named: Vec<(String, _)> = pfms
        .into_iter()
        .enumerate()
        .map(|(i, p)| (format!("f{i}"), p))
        .collect();
    let meme = write_meme(&named, Alphabet::DNA.symbols(), &bg);
    assert!(meme.starts_with("MEME version 4\n"));
    assert!(meme.contains("ALPHABET= ACGT"));
    assert!(meme.contains("MOTIF f0\nletter-probability matrix: alength= 4 w= 2 nsites= 3"));
    assert!(!meme.contains("MOTIF f1"), "a filter with no sites is left out");
}
