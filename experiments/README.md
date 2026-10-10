# bio-flodl experiments

Reproduces the evaluation of the bio-flodl application note. A separate crate (not published,
not part of the `bio-flodl` package) so the library carries no experiment dependencies.

```
cargo run --release -- gue --data <dir>/GUE --seeds 5                     # CPU
cargo run --release --features cuda -- gue --data <dir>/GUE --device cuda # GPU (CUDA libtorch)
cargo run --release -- motif --seeds 5
cargo run --release -- --help
```

Every model is trained for the same fixed number of epochs (`--epochs`, default 20) with no
model selection, so the protocol is identical across models. Each seed sets model
initialisation, dropout and batch order; the data never changes.

### `gue`: transcription-factor binding

The human transcription-factor binding task of GUE, the benchmark released with DNABERT-2
(Zhou et al., ICLR 2024): five datasets drawn from the 690 ENCODE ChIP-seq experiments, each
sequence the 101 bp around a peak centre, with GC-matched negatives. Each dataset has fixed
train (19,000 to 32,378 sequences), dev (1,000) and test (1,000) splits, about half positive.
Models train on train, the per-epoch curves follow dev, and every reported number is on test.
GUE's authors selected these five out of the 690 by excluding tasks too easy or too hard for
DNA language models, which is worth stating next to any comparison with them.

Download `GUE_v2.zip` from the link in the [DNABERT-2 README](https://github.com/MAGICS-LAB/DNABERT_2)
(Google Drive; the archive used here has SHA-256
`581ba5a69843769f4cbc34470e2fe970f8ee371280914bd66574af01ee60cd79`), extract it, and pass the
`GUE` directory as `--data`; only `GUE/tf/0` to `GUE/tf/4` are read.

### `motif`: planted-motif sanity check

Random DNA where positives carry `TATAAA`: `--n` sequences of `--len` bases, an 80/20 split
that is the same for every model and seed. Reported numbers are on the 20%.

Every run executes in its own process, so its memory figures belong to that run alone.

## Output

```
runs/gue-overview.md  MCC and AUROC per TF dataset (columns) for every model and device
runs/<dataset>/       gue-tf-0 to gue-tf-4, or motif-n<N>-len<L>
  runs.csv          one row per run (for plotting)
  table.md          mean ± sample sd per model and device
  <model>/<device>/seed-<n>/
    dashboard.html  flodl Monitor archive: every curve and resource graph, opens in a browser
    epochs.csv      per-epoch loss, accuracy, learning rate, CPU / RAM / GPU / VRAM
    predictions.csv per evaluated sample: label and positive-class probability
    roc.csv         ROC curve points (fpr, tpr)
    pr.csv          precision-recall curve points (recall, precision)
    run.json        seed, configuration, hardware, bio-flodl commit and results
```

## Reading the numbers

- CPU runs repeat exactly for a given seed. GPU runs repeat statistically, not bit for bit:
  some CUDA kernels are nondeterministic, which is why tables report the spread over seeds.
- `RSS MB` is the process's resident memory, sampled at the end of each epoch. A `cuda` build
  loads the CUDA libraries even for CPU runs (about 200 MB more), so take CPU memory figures
  from a build without `--features cuda`.
- Timings are wall time, so run timing sweeps on an otherwise idle machine.
- `GPU MB` is the peak memory allocated to tensors (`torch.cuda.max_memory_allocated`), not
  what `nvidia-smi` reports, which also counts the CUDA context and the allocator's cache.
