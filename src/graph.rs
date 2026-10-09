//! Graphs and adjacency matrices for GNN-style models and structural biology.

use crate::{Error, Result};
use std::collections::{HashMap, VecDeque};

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone)]
pub struct Graph {
    pub n: usize,
    pub directed: bool,
    pub edges: Vec<(usize, usize, f32)>,
}

impl Graph {
    pub fn new(n: usize, directed: bool) -> Self {
        Graph {
            n,
            directed,
            edges: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, w: f32) -> Result<()> {
        if u >= self.n || v >= self.n {
            return Err(Error::Invalid(format!(
                "edge ({u},{v}) out of range for {} nodes",
                self.n
            )));
        }
        self.edges.push((u, v, w));
        Ok(())
    }

    pub fn from_edges(n: usize, directed: bool, edges: &[(usize, usize)]) -> Result<Self> {
        let mut g = Graph::new(n, directed);
        for &(u, v) in edges {
            g.add_edge(u, v, 1.0)?;
        }
        Ok(g)
    }

    /// Build from a dense row-major `n x n` adjacency matrix (non-zero entries become edges).
    pub fn from_dense(adj: &[f32], n: usize, directed: bool) -> Result<Self> {
        if adj.len() != n * n {
            return Err(Error::Shape(format!(
                "expected {} entries, got {}",
                n * n,
                adj.len()
            )));
        }
        let mut g = Graph::new(n, directed);
        for i in 0..n {
            let start = if directed { 0 } else { i };
            for j in start..n {
                let w = adj[i * n + j];
                if w != 0.0 {
                    g.edges.push((i, j, w));
                }
            }
        }
        Ok(g)
    }

    /// Dense row-major adjacency matrix `[n, n]`. Undirected edges are symmetrised.
    pub fn adjacency(&self) -> Vec<f32> {
        let n = self.n;
        let mut a = vec![0.0; n * n];
        for &(u, v, w) in &self.edges {
            a[u * n + v] = w;
            if !self.directed {
                a[v * n + u] = w;
            }
        }
        a
    }

    pub fn adjacency_with_self_loops(&self) -> Vec<f32> {
        let mut a = self.adjacency();
        for i in 0..self.n {
            a[i * self.n + i] = 1.0;
        }
        a
    }

    /// Out-degree (weighted) per node.
    pub fn degrees(&self) -> Vec<f32> {
        let n = self.n;
        let a = self.adjacency();
        (0..n).map(|i| a[i * n..(i + 1) * n].iter().sum()).collect()
    }

    /// GCN propagation matrix `D^-1/2 (A + I) D^-1/2` (Kipf & Welling).
    pub fn gcn_normalized(&self) -> Vec<f32> {
        let n = self.n;
        let a = self.adjacency_with_self_loops();
        let d: Vec<f32> = (0..n)
            .map(|i| {
                let s: f32 = a[i * n..(i + 1) * n].iter().sum();
                if s > 0.0 {
                    s.powf(-0.5)
                } else {
                    0.0
                }
            })
            .collect();
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = d[i] * a[i * n + j] * d[j];
            }
        }
        out
    }

    /// Row-normalised (random-walk) adjacency `D^-1 A`.
    pub fn row_normalized(&self) -> Vec<f32> {
        let n = self.n;
        let mut a = self.adjacency();
        for i in 0..n {
            let s: f32 = a[i * n..(i + 1) * n].iter().sum();
            if s > 0.0 {
                a[i * n..(i + 1) * n].iter_mut().for_each(|x| *x /= s);
            }
        }
        a
    }

    /// Combinatorial Laplacian `D - A`.
    pub fn laplacian(&self) -> Vec<f32> {
        let n = self.n;
        let a = self.adjacency();
        let d = self.degrees();
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let aij = a[i * n + j];
                l[i * n + j] = if i == j {
                    d[i] - aij
                } else if aij == 0.0 {
                    0.0
                } else {
                    -aij
                };
            }
        }
        l
    }

    /// Additive attention mask for dense GAT: `0` where an edge (or self-loop) exists,
    /// `-1e9` elsewhere. Row-major `[n, n]`.
    pub fn attention_mask(&self) -> Vec<f32> {
        self.adjacency_with_self_loops()
            .into_iter()
            .map(|w| if w != 0.0 { 0.0 } else { -1e9 })
            .collect()
    }

    /// Adjacency list (neighbours by out-edge).
    pub fn neighbors(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.n];
        for &(u, v, _) in &self.edges {
            adj[u].push(v);
            if !self.directed {
                adj[v].push(u);
            }
        }
        adj
    }

    /// COO edge index `[2, E]` as (sources, targets), both directions for undirected graphs.
    pub fn edge_index(&self) -> (Vec<i64>, Vec<i64>) {
        let (mut s, mut t) = (Vec::new(), Vec::new());
        for &(u, v, _) in &self.edges {
            s.push(u as i64);
            t.push(v as i64);
            if !self.directed && u != v {
                s.push(v as i64);
                t.push(u as i64);
            }
        }
        (s, t)
    }

    /// Weakly connected component label per node.
    pub fn connected_components(&self) -> Vec<usize> {
        let mut adj = vec![Vec::new(); self.n];
        for &(u, v, _) in &self.edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut label = vec![usize::MAX; self.n];
        let mut c = 0;
        for s in 0..self.n {
            if label[s] != usize::MAX {
                continue;
            }
            let mut q = VecDeque::from([s]);
            label[s] = c;
            while let Some(u) = q.pop_front() {
                for &v in &adj[u] {
                    if label[v] == usize::MAX {
                        label[v] = c;
                        q.push_back(v);
                    }
                }
            }
            c += 1;
        }
        label
    }
}

/// Residue contact map → graph. `dist` is a row-major `n x n` distance matrix (Å);
/// residues closer than `cutoff` and separated by at least `min_seq_sep` in sequence are linked.
pub fn from_contact_map(dist: &[f32], n: usize, cutoff: f32, min_seq_sep: usize) -> Result<Graph> {
    if dist.len() != n * n {
        return Err(Error::Shape(format!(
            "expected {} entries, got {}",
            n * n,
            dist.len()
        )));
    }
    let mut g = Graph::new(n, false);
    for i in 0..n {
        for j in (i + 1)..n {
            if j - i >= min_seq_sep && dist[i * n + j] < cutoff {
                g.edges.push((i, j, 1.0));
            }
        }
    }
    Ok(g)
}

/// Euclidean distance matrix from `n` 3-D coordinates (e.g. C-alpha atoms).
pub fn distance_matrix(coords: &[[f32; 3]]) -> Vec<f32> {
    let n = coords.len();
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            d[i * n + j] = (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }
    d
}

/// Sequence-chain graph (i — i+1), the backbone of a protein/RNA graph.
pub fn chain(n: usize) -> Graph {
    let mut g = Graph::new(n, false);
    for i in 1..n {
        g.edges.push((i - 1, i, 1.0));
    }
    g
}

/// de Bruijn graph over (k-1)-mers: nodes are distinct (k-1)-mers, one directed edge
/// per k-mer occurrence (weight = multiplicity). Returns the node labels and the graph.
pub fn de_bruijn(seqs: &[&[u8]], k: usize) -> Result<(Vec<Vec<u8>>, Graph)> {
    if k < 2 {
        return Err(Error::Invalid("k must be >= 2".into()));
    }
    let mut ids: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut labels: Vec<Vec<u8>> = Vec::new();
    let mut weights: HashMap<(usize, usize), f32> = HashMap::new();
    let id_of = |m: &[u8], ids: &mut HashMap<Vec<u8>, usize>, labels: &mut Vec<Vec<u8>>| -> usize {
        let key = m.to_ascii_uppercase();
        *ids.entry(key.clone()).or_insert_with(|| {
            labels.push(key);
            labels.len() - 1
        })
    };
    for s in seqs {
        for w in s.windows(k) {
            let a = id_of(&w[..k - 1], &mut ids, &mut labels);
            let b = id_of(&w[1..], &mut ids, &mut labels);
            *weights.entry((a, b)).or_insert(0.0) += 1.0;
        }
    }
    let mut g = Graph::new(labels.len(), true);
    let mut es: Vec<_> = weights.into_iter().collect();
    es.sort_by_key(|&((a, b), _)| (a, b));
    g.edges = es.into_iter().map(|((a, b), w)| (a, b, w)).collect();
    Ok((labels, g))
}
