# bio-flodl

Bioinformatics data layer **and model zoo** for [flodl](https://docs.rs/flodl) (Rust deep learning on libtorch).

## What's in it

| Area | Contents | Verified |
|---|---|---|
| Core (pure Rust, no libtorch) | FASTA parsing and fixed-length cropping/padding; DNA/RNA/protein alphabets; one-hot (strict/lenient/labels/batch+mask, channels-first); k-mers; reverse complement; graphs, dense/GCN-normalised/row-normalised adjacency, Laplacian, attention mask, contact maps, de Bruijn; datasets, batching, seeded RNG, accuracy / confusion matrix / macro-F1 / MCC / AUROC / AUPRC; synthetic motif and graph tasks; sinusoidal positions | 16 tests pass |
| `models::cnn` | `MotifCnn` (parallel multi-width Conv1d motif scanners, global max-pool, filter extraction), `DilatedResCnn` (residual dilated Conv1d, exponential receptive field) | type-checked |
| `models::rnn` | `BiRnnClassifier` (bidirectional LSTM or GRU) | type-checked |
| `models::transformer` | `TransformerClassifier` (pre-LN encoder, MHA, sinusoidal positions, optional attention mask) | type-checked |
| `models::gnn` | `GcnLayer`/`GcnNet`, `GatLayer`/`GatNet` (multi-head, dense), node or graph-mean readout | type-checked |
| `models::mlp` | `FlatMlp` baseline | type-checked |
| `models::compose` | `Sequential`, `Residual`, `Lambda` (relu/gelu/global pooling/flatten/transpose), `ConcatBranches`: assemble **any** flodl layer into a trainable model | type-checked |
| `prelude` | `use bio_flodl::prelude::*;` gives all of flodl plus this crate's models and trainer | type-checked |
| `train` | `fit` (any `Module` + any `Optimizer`, grad clipping, LR decay, early stopping), `evaluate`, `predict_proba`, `predict_probs`, `report` (accuracy, macro-F1, MCC, AUROC, AUPRC) | type-checked |

**"Type-checked" means** it compiles against a stub that mirrors the flodl 0.8 signatures read from
docs.rs (`tools/typecheck.sh`). It has **not been run** against real libtorch, so numerical behaviour and
shape handling are unverified. See *Assumptions to check* below.

## Everything in flodl is available through the prelude

`bio-flodl` does not reimplement flodl's layers. With the `flodl` feature on,

```rust
use bio_flodl::prelude::*;
```

is a glob re-export of the whole flodl crate root (`pub use flodl::*`) **plus** this crate's models,
containers and trainer. So whatever flodl exports at its top level is usable here, and the set stays in
sync with the flodl version you build against.

### Component catalogue (flodl 0.8, as listed on docs.rs)

| Category | Components |
|---|---|
| **Convolution** | `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` |
| **Pooling / resampling** | `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `AdaptiveMaxPool2d`, `AdaptiveAvgPool2d`, `Upsample`, `Unfold`, `Fold`, `PixelShuffle`, `PixelUnshuffle` |
| **Normalisation** | `BatchNorm`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `InstanceNorm`, `RMSNorm` |
| **Dropout** | `Dropout`, `Dropout2d`, `AlphaDropout` |
| **Embeddings and sequence layers** | `Embedding`, `EmbeddingBag`, `LSTM`, `GRU`, `LSTMCell`, `GRUCell`, `MultiheadAttention`, `RotaryEmbedding`, `Bilinear` |
| **Dense / padding** | `Linear`, `ZeroPad2d`, `ReflectionPad2d` |
| **Activations** | `ReLU`, `GELU`, `SiLU`, `Mish`, `SELU`, `ELU`, `LeakyReLU`, `PReLU`, `Softmax`, `LogSoftmax`, `SwiGLU` and others |
| **Losses** | cross-entropy, BCE (incl. with logits), MSE, L1, smooth-L1, focal, CTC, KL-divergence, triplet and others |
| **Optimizers** | `Adam`, `AdamW`, `SGD`, `RMSprop`, `NAdam`, `RAdam`, `Adagrad` |
| **LR schedulers** | cosine, one-cycle, step, multi-step, exponential, plateau, warmup, cyclic |

### Limits of "everything is in bio-flodl"

1. **Only through the prelude, only with `--features flodl`.** `bio_flodl::Conv2d` at the crate root does not
   exist. Without the feature, `bio-flodl` does not depend on flodl.
2. **Only flodl's top-level exports.** Items that flodl keeps in a submodule without re-exporting at the root
   must be imported from `flodl` directly (add `flodl` to your own `Cargo.toml`; it is the same crate).
3. **On a name clash, bio-flodl wins.** This crate's types (`MotifCnn`, `Sequential`, `Readout`, `fit`, `adam`, ...)
   are explicit re-exports and shadow glob-imported flodl names. I know of no actual clash but have not
   compared every flodl name against mine.
4. flodl itself needs libtorch and Rust 1.85+.

### Using any component in a trainable model

`models::compose` wires any flodl layer into something `train::fit` can train:

| Container | What it does |
|---|---|
| `Sequential::new().push(layer)...` | run layers in order; accepts any `Module` |
| `Residual::new(layer)` | `x + layer(x)` (shape-preserving) |
| `Lambda::{relu, gelu, tanh, sigmoid, flatten, global_max_pool1d, global_avg_pool1d, transpose(a, b)}` | parameter-free glue between layers |
| `ConcatBranches::new(dim).branch(a).branch(b)` | parallel branches concatenated along `dim` |

Sketch (constructors shown for `Conv1d`, `Dropout`, `Linear` are the verified ones; substitute your own
`MaxPool1d`, `BatchNorm`, `Conv2d`, ... after checking their signatures):

```rust
use bio_flodl::prelude::*;

let model = Sequential::new()
    .push(Conv1d::new(4, 64, 9)?)      // [B, 4, L] -> [B, 64, L-8]
    .push(Lambda::relu())
    .push(Dropout::new(0.2))
    .push(Lambda::global_max_pool1d()) // [B, 64]
    .push(Linear::new(64, 2)?);

let mut opt = adam(&model, 1e-3);
fit(&model, &mut opt, &train, Some(&val), &TrainConfig::default())?;
```

Two things to know when composing: a `Lambda::transpose(1, 2)` converts between the CNN layout
`[B, C, L]` and the RNN/Transformer layout `[B, L, C]`; and `Dropout` only switches between train and eval
if the container forwards `set_training`, which `Sequential`, `Residual` and `ConcatBranches` do.

## Examples and benchmarks (`examples/`)

Run with `cargo run --release --example <name>`; add `--features flodl` for the model examples
(needs libtorch and Rust 1.85+).

| File | Needs flodl | What it does | Status |
|---|---|---|---|
| `encode_tour.rs` | no | one-hot (strict/lenient), k-mers, reverse complement, padded batches, channels-first, protein, positions | **run** |
| `graph_tour.rs` | no | adjacency, GCN/row-normalised adjacency, Laplacian, edge index, components, contact map from coordinates, de Bruijn | **run** |
| `metrics_demo.rs` | no | datasets, splits, accuracy / macro-F1 / MCC / AUROC / AUPRC | **run** |
| `bench_core.rs` | no | throughput of the data layer | **run** (results below) |
| `motif_cnn.rs` | yes | multi-width motif CNN on a planted motif | type-checked |
| `dilated_cnn.rs` | yes | residual dilated CNN | type-checked |
| `birnn.rs` | yes | BiLSTM and BiGRU | type-checked |
| `transformer.rs` | yes | Transformer encoder | type-checked |
| `gcn_graph.rs` | yes | GCN graph classification on a contact graph | type-checked |
| `gat_graph.rs` | yes | multi-head GAT, same task | type-checked |
| `compose_sequential.rs` | yes | custom model from `Sequential` / `ConcatBranches` / `Lambda` using verified layers | type-checked |
| `bench_models.rs` | yes | train and inference throughput plus parameter count for every bundled model | type-checked, **not run** |
| `tf_binding.rs` | yes | runs every sequence model on a positive/negative FASTA pair and prints accuracy, F1, MCC, AUROC, AUPRC, time | type-checked, **not run, needs your data** |

"Type-checked" means compiled against the flodl signature stub (`tools/typecheck.sh`); it does not mean executed.
The synthetic tasks (planted motif, graph classification) are smoke tests: a model that cannot learn them has a bug,
but good scores on them say nothing about real biological performance.

### Core benchmark results (measured)

`cargo run --release --example bench_core`, best of 5, on a **1 vCPU Intel Xeon @ 2.80 GHz sandbox**.
Absolute numbers will differ on your hardware; rerun it for anything you intend to publish.

| operation | time | throughput |
|---|---|---|
| `one_hot` (2000 x 1000 bp, one by one) | 23.4 ms | 85.4 M bases/s |
| `one_hot_batch` + `channels_first` (2000 x 1000) | 52.8 ms | 37.9 M bases/s |
| `kmer_counts` k=6 (2 Mbp) | 91.4 ms | 21.9 M bases/s |
| `reverse_complement` (2 Mbp) | 15.9 ms | 125.9 M bases/s |
| `adjacency` (n=500) | 0.022 ms | *allocation only, see note* |
| `gcn_normalized` (n=500) | 0.68 ms | 367.5 M cells/s |
| `laplacian` (n=500) | 0.64 ms | 389.9 M cells/s |
| `distance_matrix` + contact graph (n=500) | 0.33 ms | 749.9 M cells/s |
| `de_bruijn` k=21 (200 reads x 1000 bp) | 202.5 ms | 1.0 M bases/s |
| `auroc` (1 M scores) | 235.1 ms | 4.3 M scores/s |
| `average_precision` (1 M scores) | 219.5 ms | 4.6 M scores/s |
| `Dataset::split` + batch gather (2000 x 1000) | 8.6 ms | 232.4 M bases/s |

Notes: the `adjacency` figure is dominated by a lazily zeroed allocation, so its apparent throughput is not
meaningful. `de_bruijn` is the slowest operation because it keys a hash map on freshly allocated `Vec<u8>`
(k-1)-mers; a 2-bit-packed key would be much faster and is an obvious optimisation. Metric functions sort,
so they are O(n log n).

### Model benchmarks (not yet measured)

`bench_models` and `tf_binding` are written but **have never been executed**, so this README contains no model
results. Once you have libtorch, the intended protocol is:

1. `bench_models` -> parameters, ms/step and samples/s per model (CPU tensors only as written; a GPU run
   needs the model and inputs moved to the device).
2. Implement the same architectures and batch size in PyTorch and time them on the same machine for a fair comparison.
3. `tf_binding pos.fa neg.fa` on a real dataset (for example ENCODE ChIP-seq peaks versus matched background)
   with several `--seed` values, and report mean and spread.

## Layouts

* CNNs: `[batch, alphabet, length]` (use `Dataset::from_sequences(.., channels_first = true)`)
* RNN / Transformer / MLP: `[batch, length, alphabet]`
* GNNs: `[nodes, features]` or `[batch, nodes, features]` sharing one adjacency

## Quick example

```rust
use bio_flodl::{alphabet::Alphabet, data::Dataset, models::cnn::*, train::*};

let ds = Dataset::from_sequences(&seqs, &labels, &Alphabet::DNA, 100, true)?;
let (train, val) = ds.split(0.2, 1);
let model = MotifCnn::new(&MotifCnnConfig { in_channels: 4, filters: 32, kernel_sizes: vec![6, 9, 12], dropout: 0.2, classes: 2 })?;
let mut opt = adam(&model, 1e-3);
fit(&model, &mut opt, &train, Some(&val), &TrainConfig::default())?;
```

Full version: `examples/motif_cnn.rs`.

```
cargo test                                    # core, no libtorch
tools/typecheck.sh                            # flodl-feature code vs. signature stub, no libtorch
cargo run --release --features flodl --example motif_cnn   # needs libtorch + Rust 1.85
```

Gaurav Sablok \
gsablok@proton.me
