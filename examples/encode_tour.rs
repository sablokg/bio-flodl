//! Sequence encodings. No libtorch needed: `cargo run --example encode_tour`

use bio_flodl::alphabet::{gc_content, reverse_complement, Alphabet};
use bio_flodl::encode::*;

fn main() {
    let dna = b"ACGTNACGT";

    // strict mode rejects ambiguity codes, lenient maps them to an all-zero row
    println!("strict: {:?}", one_hot(dna, &Alphabet::DNA).map(|_| ()));
    let lenient = one_hot_lenient(dna, &Alphabet::DNA);
    println!("lenient shape {:?}; row of 'N' = {:?}", lenient.shape(), &lenient.data[16..20]);

    println!("reverse complement of ACGTT = {}", String::from_utf8(reverse_complement(b"ACGTT")).unwrap());
    println!("GC content of GGCCAT = {:.3}", gc_content(b"GGCCAT"));

    // k-mers
    let k2 = kmer_frequencies(b"ACGTACGTAA", 2, &Alphabet::DNA).unwrap();
    println!("2-mer frequencies (16 dims), AC = {:.3}", k2[1]);

    // batching with padding + mask, then the Conv1d layout
    let batch = one_hot_batch(&[b"ACGT", b"AC"], &Alphabet::DNA, None);
    println!("batch shape {:?}, mask {:?}", batch.shape(), batch.mask);
    println!("channels-first has {} values, shape [n, alphabet, len]", batch.channels_first().len());

    // proteins use the same API
    let protein = one_hot(b"MKTAYIAKQR", &Alphabet::PROTEIN).map(|o| o.shape());
    println!("protein one-hot shape {:?}", protein.ok());

    // class labels
    let labels = one_hot_labels(&[0, 2, 1], 3).unwrap();
    println!("label one-hot {:?}", labels.data);

    // sinusoidal positions for Transformers
    let pe = sinusoidal_positions(3, 4);
    println!("position 1 encoding {:?}", &pe[4..8]);
}
