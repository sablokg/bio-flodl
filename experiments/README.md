# bio-flodl experiments

Reproduces the evaluation of the bio-flodl application note. A separate crate (not published,
not part of the `bio-flodl` package) so the library carries no experiment dependencies.

```
cargo run --release -- motif --seeds 5                                    # CPU
cargo run --release --features cuda -- motif --seeds 5 --device cuda      # GPU (CUDA libtorch)
cargo run --release -- motif --help
```

`motif` trains every sequence model on the planted-motif task (random DNA, positives carry
`TATAAA`; `--n` sequences of `--len` bases, an 80/20 split that is the same for every model and
seed). The data is fixed; each seed sets model initialisation, dropout and batch order.

Every run executes in its own process, so its memory figures belong to that run alone.

## Output

```
runs/<task>/
  runs.csv          one row per run (for plotting)
  table.md          mean ± sample sd per model and device
  <model>/<device>/seed-<n>/
    dashboard.html  flodl Monitor archive: every curve and resource graph, opens in a browser
    epochs.csv      per-epoch loss, accuracy, learning rate, CPU / RAM / GPU / VRAM
    predictions.csv per validation sample: label and positive-class probability
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
- `GPU MB` is the peak memory allocated to tensors (`torch.cuda.max_memory_allocated`), not
  what `nvidia-smi` reports, which also counts the CUDA context and the allocator's cache.
