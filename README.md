# flodl-bio

Bioinformatics data layer **and model zoo** for [flodl](https://docs.rs/flodl) (Rust deep learning on libtorch).

flodl-bio is a community crate built on flodl, developed and maintained by Gaurav Sablok.

## What's in it

| Area | Contents | Verified |
|---|---|---|
| Core (pure Rust, no libtorch) | FASTA parsing and fixed-length cropping/padding; DNA/RNA/protein alphabets; one-hot (strict/lenient/labels/batch+mask, channels-first); k-mers; reverse complement; graphs, dense/GCN-normalised/row-normalised adjacency, Laplacian, attention mask, contact maps, de Bruijn; datasets, batching, seeded RNG, accuracy / confusion matrix / macro-F1 / MCC / AUROC / AUPRC, ROC and PR curve points; motifs from convolutional filters (frequency matrices, MEME files for TOMTOM); synthetic motif and graph tasks; sinusoidal positions | 18 tests pass |
| `models::cnn` | `MotifCnn` (parallel multi-width Conv1d motif scanners, global max-pool, filter extraction, activation scans), `DilatedResCnn` (residual dilated Conv1d, exponential receptive field) | tested, CPU + GPU |
| `models::rnn` | `BiRnnClassifier` (bidirectional LSTM or GRU) | tested, CPU + GPU |
| `models::transformer` | `TransformerClassifier` (pre-LN encoder, MHA, sinusoidal positions, optional attention mask) | tested, CPU + GPU |
| `models::mlp` | `FlatMlp` baseline | tested, CPU + GPU |
| `models::compose` | `Sequential`, `Residual`, `Lambda` (relu/gelu/global pooling/flatten/transpose), `ConcatBranches`: assemble **any** flodl layer into a trainable model | tested, CPU + GPU |
| `prelude` | `use flodl_bio::prelude::*;` gives all of flodl plus this crate's models and trainer | tested |
| `train` | `fit` (any `Module` + any `Optimizer`, grad clipping, LR decay, early stopping), `fit_with` (same, with a per-epoch callback), `evaluate`, `predict_proba`, `predict_probs`, `report` (accuracy, macro-F1, MCC, AUROC, AUPRC) | tested, CPU + GPU |

**"Tested, CPU + GPU" means** `tests/models.rs` trains and evaluates every bundled model, and a container
holding `BatchNorm`, against flodl 0.8.0 (libtorch 2.10) on the CPU and, with `--features cuda`, on the GPU.
`tools/typecheck.sh` still checks the flodl-feature code against a signature stub when libtorch is not
installed.

## Everything in flodl is available through the prelude

`flodl-bio` does not reimplement flodl's layers. With the `flodl` feature on,

```rust
use flodl_bio::prelude::*;
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

### Limits of "everything is in flodl-bio"

1. **Only through the prelude, only with `--features flodl`.** `flodl_bio::Conv2d` at the crate root does not
   exist. Without the feature, `flodl-bio` does not depend on flodl.
2. **Only flodl's top-level exports.** Items that flodl keeps in a submodule without re-exporting at the root
   must be imported from `flodl` directly (add `flodl` to your own `Cargo.toml`; it is the same crate).
3. **On a name clash, flodl-bio wins.** This crate's types (`MotifCnn`, `Sequential`, `fit`, `adam`, ...)
   are explicit re-exports and shadow glob-imported flodl names. I know of no actual clash but have not
   compared every flodl name against mine.
4. flodl itself needs libtorch and Rust 1.91+.

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
use flodl_bio::prelude::*;

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
(needs libtorch and Rust 1.91+). The examples train on the CPU; see *GPU* below to move a model.

| File | Needs flodl | What it does | Status |
|---|---|---|---|
| `encode_tour.rs` | no | one-hot (strict/lenient), k-mers, reverse complement, padded batches, channels-first, protein, positions | **run** |
| `graph_tour.rs` | no | adjacency, GCN/row-normalised adjacency, Laplacian, edge index, components, contact map from coordinates, de Bruijn | **run** |
| `metrics_demo.rs` | no | datasets, splits, accuracy / macro-F1 / MCC / AUROC / AUPRC | **run** |
| `bench_core.rs` | no | throughput of the data layer | **run** (results below) |
| `motif_cnn.rs` | yes | multi-width motif CNN on a planted motif | **run** |
| `dilated_cnn.rs` | yes | residual dilated CNN | **run** |
| `birnn.rs` | yes | BiLSTM and BiGRU | **run** |
| `transformer.rs` | yes | Transformer encoder | **run** |
| `compose_sequential.rs` | yes | custom model from `Sequential` / `ConcatBranches` / `Lambda` using verified layers | **run** |
| `bench_models.rs` | yes | train and inference throughput plus parameter count for every bundled model | **run** |
| `tf_binding.rs` | yes | runs every sequence model on a positive/negative FASTA pair and prints accuracy, F1, MCC, AUROC, AUPRC, time | builds, **needs your data** |

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

## Layouts

* CNNs: `[batch, alphabet, length]` (use `Dataset::from_sequences(.., channels_first = true)`)
* RNN / Transformer / MLP: `[batch, length, alphabet]`

## Quick example

```rust
use flodl_bio::{alphabet::Alphabet, data::Dataset, models::cnn::*, train::*};

let ds = Dataset::from_sequences(&seqs, &labels, &Alphabet::DNA, 100, true)?;
let (train, val) = ds.split(0.2, 1);
let model = MotifCnn::new(&MotifCnnConfig { in_channels: 4, filters: 32, kernel_sizes: vec![6, 9, 12], dropout: 0.2, classes: 2 })?;
let mut opt = adam(&model, 1e-3);
fit(&model, &mut opt, &train, Some(&val), &TrainConfig::default())?;
```

Full version: `examples/motif_cnn.rs`.

## GPU

Build against a CUDA libtorch with `--features cuda`, then move the model before creating its optimizer.
`fit`, `evaluate`, `predict_probs` and `report` build their batches on the device of the model's parameters,
and the transformer's positional table follows the input, so nothing else changes:

```rust
let model = MotifCnn::new(&cfg)?;
model.move_to_device(Device::CUDA(0));
let mut opt = adam(&model, 1e-3);
fit(&model, &mut opt, &train, Some(&val), &TrainConfig::default())?;
```

## Commands

```
cargo test                                               # core, no libtorch
cargo test --features flodl                              # core + model tier, CPU
cargo test --features cuda --test models -- --ignored    # model tier on the GPU
tools/typecheck.sh                                       # flodl-feature code vs. signature stub, no libtorch
cargo run --release --features flodl --example motif_cnn # needs libtorch + Rust 1.91
```

Gaurav Sablok \
gsablok@proton.me
