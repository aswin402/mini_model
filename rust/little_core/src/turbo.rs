//! Fast Walsh-Hadamard Transform (FWHT), 1-bit/4-bit TurboQuant Vector Quantization & Index.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct QuantizedVectorRust {
    pub id: String,
    pub vector_1bit: [u64; 8],
    pub vector_4bit: Vec<u8>,
    pub scale: f32,
    pub norm: f32,
    pub metadata_json: String,
}

#[derive(Serialize)]
pub struct SearchResultRust {
    pub id: String,
    pub score: f32,
    pub hamming_distance: u32,
    pub metadata: serde_json::Value,
}

pub struct TurboIndexRust {
    pub dimension: usize,
    pub signs: Vec<f32>,
    pub scale_factor: f32,
    pub vectors: Vec<QuantizedVectorRust>,
    pub id_to_index: HashMap<String, usize>,
}

impl TurboIndexRust {
    pub fn new(dimension: usize, signs: &[f32]) -> Result<Self, &'static str> {
        if dimension != 512 || signs.len() != 512 {
            return Err("TurboQuant supports dimension 512 only");
        }
        if signs
            .iter()
            .any(|sign| !sign.is_finite() || sign.abs() != 1.0)
        {
            return Err("rotation signs must be +1 or -1");
        }
        let scale_factor = 1.0 / (dimension as f32).sqrt();
        Ok(Self {
            dimension,
            signs: signs.to_vec(),
            scale_factor,
            vectors: Vec::new(),
            id_to_index: HashMap::new(),
        })
    }

    pub fn rotate(&self, vector: &[f32], out: &mut [f32]) {
        assert_eq!(vector.len(), self.dimension);
        assert_eq!(out.len(), self.dimension);

        // 1. Sign flip D * x
        for i in 0..self.dimension {
            out[i] = vector[i] * self.signs[i];
        }

        // 2. In-place Fast Walsh-Hadamard Transform H * (D * x)
        let mut h = 1;
        let n = self.dimension;
        while h < n {
            let step = h * 2;
            let mut i = 0;
            while i < n {
                for j in i..(i + h) {
                    let u = out[j];
                    let v = out[j + h];
                    out[j] = u + v;
                    out[j + h] = u - v;
                }
                i += step;
            }
            h = step;
        }

        // 3. Orthogonal normalization (1 / sqrt(d))
        for x in out.iter_mut() {
            *x *= self.scale_factor;
        }
    }

    pub fn quantize_1bit(&self, rotated: &[f32]) -> [u64; 8] {
        let mut ints = [0u64; 8];
        for chunk in 0..8 {
            let mut val = 0u64;
            let offset = chunk * 64;
            for bit in 0..64 {
                if rotated[offset + bit] >= 0.0 {
                    val |= 1u64 << bit;
                }
            }
            ints[chunk] = val;
        }
        ints
    }

    pub fn quantize_4bit(&self, rotated: &[f32]) -> (Vec<u8>, f32) {
        let mut alpha = 0.0f32;
        for &v in rotated {
            let abs_v = v.abs();
            if abs_v > alpha {
                alpha = abs_v;
            }
        }
        if alpha < 1e-12 {
            alpha = 1.0;
        }

        let inv_scale = 7.0 / alpha;
        let mut packed = vec![0u8; self.dimension / 2];
        for i in (0..self.dimension).step_by(2) {
            let n0 = ((rotated[i] * inv_scale).round() as i32).clamp(-8, 7) + 8;
            let n1 = ((rotated[i + 1] * inv_scale).round() as i32).clamp(-8, 7) + 8;
            packed[i / 2] = ((n0 as u8) << 4) | (n1 as u8);
        }
        (packed, alpha)
    }

    pub fn add(&mut self, id: &str, vector: &[f32], metadata_json: &str) {
        let norm_sq: f32 = vector.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        let mut normalized = vec![0.0f32; self.dimension];
        if norm > 1e-12 {
            let inv_norm = 1.0 / norm;
            for i in 0..self.dimension {
                normalized[i] = vector[i] * inv_norm;
            }
        } else {
            normalized.copy_from_slice(vector);
        }

        let mut rotated = vec![0.0f32; self.dimension];
        self.rotate(&normalized, &mut rotated);
        let v1 = self.quantize_1bit(&rotated);
        let (v4, scale) = self.quantize_4bit(&rotated);

        let qv = QuantizedVectorRust {
            id: id.to_string(),
            vector_1bit: v1,
            vector_4bit: v4,
            scale,
            norm: if norm > 1e-12 { norm } else { 1.0 },
            metadata_json: metadata_json.to_string(),
        };

        if let Some(&existing_idx) = self.id_to_index.get(id) {
            self.vectors[existing_idx] = qv;
        } else {
            let idx = self.vectors.len();
            self.id_to_index.insert(id.to_string(), idx);
            self.vectors.push(qv);
        }
    }

    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(idx) = self.id_to_index.remove(id) {
            self.vectors.swap_remove(idx);
            if idx < self.vectors.len() {
                let swapped_id = self.vectors[idx].id.clone();
                self.id_to_index.insert(swapped_id, idx);
            }
            true
        } else {
            false
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        threshold: f32,
        coarse_multiplier: usize,
    ) -> Vec<SearchResultRust> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        let mut normalized_q = vec![0.0f32; self.dimension];
        if norm > 1e-12 {
            let inv_norm = 1.0 / norm;
            for i in 0..self.dimension {
                normalized_q[i] = query[i] * inv_norm;
            }
        } else {
            normalized_q.copy_from_slice(query);
        }

        let mut q_rot = vec![0.0f32; self.dimension];
        self.rotate(&normalized_q, &mut q_rot);
        let q_1bit = self.quantize_1bit(&q_rot);

        let total_vecs = self.vectors.len();
        let coarse_pool_size = (top_k.saturating_mul(coarse_multiplier))
            .min(total_vecs)
            .max(1);

        let mut coarse_scores: Vec<(u32, usize)> = Vec::with_capacity(total_vecs);
        for (idx, qv) in self.vectors.iter().enumerate() {
            let mut dist = 0u32;
            for k in 0..8 {
                dist += (qv.vector_1bit[k] ^ q_1bit[k]).count_ones();
            }
            coarse_scores.push((dist, idx));
        }

        if coarse_scores.len() > coarse_pool_size {
            coarse_scores.select_nth_unstable_by_key(coarse_pool_size, |&(d, _)| d);
            coarse_scores.truncate(coarse_pool_size);
        }
        coarse_scores.sort_unstable_by_key(|&(d, _)| d);

        let mut results: Vec<SearchResultRust> = Vec::with_capacity(coarse_scores.len());
        for (dist, idx) in coarse_scores {
            let qv = &self.vectors[idx];
            let mut accum = 0.0f32;
            let step = qv.scale / 7.0;

            for i in (0..self.dimension).step_by(2) {
                let b = qv.vector_4bit[i / 2];
                let n0 = ((b >> 4) as i32) - 8;
                let n1 = ((b & 0x0F) as i32) - 8;
                accum += (n0 as f32) * q_rot[i] + (n1 as f32) * q_rot[i + 1];
            }
            let score = accum * step;

            if score >= threshold {
                let parsed_meta: serde_json::Value =
                    serde_json::from_str(&qv.metadata_json).unwrap_or(serde_json::json!({}));
                results.push(SearchResultRust {
                    id: qv.id.clone(),
                    score: (score * 10000.0).round() / 10000.0,
                    hamming_distance: dist,
                    metadata: parsed_meta,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }
}
