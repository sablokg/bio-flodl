#!/usr/bin/env sh
# Type-checks the `flodl`-feature code against a signature-only stub of flodl 0.8,
# so you can catch type/borrow/arity errors without libtorch. It does NOT run anything.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
cp -r "$ROOT"/. "$TMP/"
rm -rf "$TMP/target" "$TMP/tools/flodl-stub/target"
printf '\n[patch.crates-io]\nflodl = { path = "%s/tools/flodl-stub" }\n' "$TMP" >> "$TMP/Cargo.toml"
( cd "$TMP" && cargo check --features flodl --all-targets )
