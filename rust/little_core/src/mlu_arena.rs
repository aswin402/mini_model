//! Rust-Native 360° Radial Liquid-Resonance Network (`SpiderWebArena` & `MluCell`).
//!
//! Replaces traditional Transformer self-attention and dense FFN layers with:
//! 1. Wave 1: LFM-2.5 Gated Short-Convolution (k=3) + Continuous-Time Liquid (`CfC`) state update.
//! 2. Wave 2: 360° 6-Phase Radial Resonance (`Taxonomy`, `Mereology`, `Attributes`,
//!    `Dynamics`, `Procedural`, `Episodic`).
//! 3. Wave 3: Local JEPA Predictive-Coding + 3-Factor Verification-Guided Plasticity
//!    and 5-Brake Fractal Mitosis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const MLU_DIM: usize = 64;
pub const NUM_SPOKES: usize = 6;

pub const SPOKE_NAMES: [&str; NUM_SPOKES] = [
    "taxonomy",
    "mereology",
    "attributes",
    "dynamics",
    "procedural",
    "episodic",
];

pub const DEFAULT_SUBCORES: [&str; NUM_SPOKES] = [
    "SUBCORE_TAXONOMY",
    "SUBCORE_MEREOLOGY",
    "SUBCORE_ATTRIBUTES",
    "SUBCORE_DYNAMICS",
    "SUBCORE_PROCEDURAL",
    "SUBCORE_EPISODIC",
];

#[inline]
fn fast_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x.min(30.0)).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.max(-30.0).exp();
        z / (1.0 + z)
    }
}

#[inline]
fn fast_silu(x: f32) -> f32 {
    x * fast_sigmoid(x)
}

#[inline]
fn normalize_in_place(v: &mut [f32; MLU_DIM]) {
    let mut sum_sq: f32 = 0.0;
    for &val in v.iter() {
        sum_sq += val * val;
    }
    let norm = sum_sq.sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for val in v.iter_mut() {
            *val *= inv;
        }
    }
}

#[inline]
fn dot_product(a: &[f32; MLU_DIM], b: &[f32; MLU_DIM]) -> f32 {
    let mut acc: f32 = 0.0;
    for i in 0..MLU_DIM {
        acc += a[i] * b[i];
    }
    acc
}

/// Deterministic unit-norm seed vector from a string key (FNV-1a + trigonometric harmonic phase).
pub fn deterministic_seed_vec(key: &str) -> [f32; MLU_DIM] {
    let clean = key.trim().to_uppercase().replace(' ', "_");
    let bytes = clean.as_bytes();
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }

    let mut out = [0.0f32; MLU_DIM];
    let mut state = h;
    for (i, slot) in out.iter_mut().enumerate() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let mixed = state.wrapping_mul(0x2545F4914F6CDD1D);
        let unit = ((mixed & 0xFFFF) as f32) / 65535.0;
        let angle = (unit * 2.0 - 1.0) * std::f32::consts::PI;
        *slot = (angle * (1.0 + (i as f32) * 0.125)).sin();
    }
    normalize_in_place(&mut out);
    out
}

/// 360° Angular Harmonic Phase Basis Vector for Spoke `k` (k = 0..6, 60° = 2πk/6 phase intervals).
pub fn harmonic_spoke_basis(spoke_idx: usize) -> [f32; MLU_DIM] {
    let theta = (spoke_idx as f32) * (2.0 * std::f32::consts::PI / (NUM_SPOKES as f32));
    let mut out = [0.0f32; MLU_DIM];
    for (d, slot) in out.iter_mut().enumerate() {
        let harmonic = 1.0 + ((d % NUM_SPOKES) as f32) * 0.5;
        let dim_phase = (d as f32) * std::f32::consts::PI / (MLU_DIM as f32);
        *slot = (theta * harmonic + dim_phase).cos();
    }
    normalize_in_place(&mut out);
    out
}

/// Pure mathematical 360° angular spoke router: projects the coupled (relation, target) wave
/// onto the 6 orthogonal harmonic phase basis vectors and returns argmax_k.
pub fn route_phase_spoke_idx(rel_vec: &[f32; MLU_DIM], target_vec: &[f32; MLU_DIM]) -> usize {
    let mut coupled = [0.0f32; MLU_DIM];
    for i in 0..MLU_DIM {
        coupled[i] = rel_vec[i] + 0.35 * target_vec[i];
    }
    normalize_in_place(&mut coupled);

    let mut best_k: usize = 0;
    let mut best_proj: f32 = -1e9;
    for k in 0..NUM_SPOKES {
        let basis = harmonic_spoke_basis(k);
        let proj = dot_product(&basis, &coupled);
        if proj > best_proj {
            best_proj = proj;
            best_k = k;
        }
    }
    best_k
}

/// Phase 11: Pure Shannon character entropy H(w) = -sum p(c) log2 p(c) in bits/char.
pub fn shannon_char_entropy(token: &str) -> f32 {
    let chars: Vec<char> = token.chars().collect();
    let n = chars.len();
    if n <= 1 {
        return 0.0;
    }
    let mut counts: HashMap<char, usize> = HashMap::new();
    for ch in chars {
        *counts.entry(ch).or_insert(0) += 1;
    }
    let inv_n = 1.0 / (n as f32);
    let mut h: f32 = 0.0;
    for &c in counts.values() {
        let p = (c as f32) * inv_n;
        h -= p * p.log2();
    }
    h
}

/// Phase 11: Information-theoretic morphological salience S(w) = H(w) * ln(1 + |w|).
pub fn token_information_salience(token: &str) -> f32 {
    if token.is_empty() {
        return 0.0;
    }
    let h = shannon_char_entropy(token);
    let len_factor = (1.0 + (token.chars().count() as f32)).ln();
    let compound_bonus = if token.contains('_') { 2.0 } else { 0.0 };
    h * len_factor + compound_bonus
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpokeSynapse {
    pub target_id: u32,
    pub target_name: String,
    pub predicate: String,
    pub affinity: f32,
    pub pos_evidence: f32,
    pub neg_evidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpokePredictionDto {
    pub source_concept: String,
    pub spoke: String,
    pub predicate: String,
    pub target_concept: String,
    pub neural_score: f32,
    pub nars_confidence: f32,
    pub epistemic_entropy: f32,
    pub combined_confidence: f32,
    pub verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MluCell {
    pub id: u32,
    pub concept_id: String,
    pub role: String, // "core" | "subcore" | "leaf"
    pub parent_hub: Option<String>,
    pub anchor_vec: Vec<f32>,
    pub hidden_state: Vec<f32>,
    pub conv_k1: Vec<f32>,
    pub conv_k2: Vec<f32>,
    pub conv_k3: Vec<f32>,
    pub gate_weights: Vec<f32>,
    pub tau_weights: Vec<f32>,
    pub spoke_weights: Vec<Vec<f32>>,
    pub spoke_biases: Vec<f32>,
    pub spoke_synapses: Vec<Vec<SpokeSynapse>>,
    pub token_transitions: HashMap<String, HashMap<String, f32>>,
    pub activation_count: u64,
    pub verified_hits: f32,
    pub refuted_hits: f32,
    pub cumulative_surprise: f32,
    pub last_surprise: f32,
}

impl MluCell {
    pub fn new(id: u32, concept_id: &str, role: &str, parent_hub: Option<String>) -> Self {
        let clean = concept_id.trim().to_uppercase().replace(' ', "_");
        let seed = deterministic_seed_vec(&clean);
        let gate = deterministic_seed_vec(&format!("GATE::{clean}"));
        let mut spoke_weights = Vec::with_capacity(NUM_SPOKES);
        for k in 0..NUM_SPOKES {
            let mut basis = harmonic_spoke_basis(k);
            for i in 0..MLU_DIM {
                basis[i] = 0.75 * basis[i] + 0.25 * seed[i];
            }
            normalize_in_place(&mut basis);
            spoke_weights.push(basis.to_vec());
        }
        Self {
            id,
            concept_id: clean,
            role: role.to_string(),
            parent_hub,
            anchor_vec: seed.to_vec(),
            hidden_state: seed.to_vec(),
            conv_k1: vec![0.60; MLU_DIM],
            conv_k2: vec![0.25; MLU_DIM],
            conv_k3: vec![0.15; MLU_DIM],
            gate_weights: gate.to_vec(),
            tau_weights: vec![0.15; MLU_DIM],
            spoke_weights,
            spoke_biases: vec![0.0; NUM_SPOKES],
            spoke_synapses: vec![Vec::new(); NUM_SPOKES],
            token_transitions: HashMap::new(),
            activation_count: 0,
            verified_hits: 0.0,
            refuted_hits: 0.0,
            cumulative_surprise: 0.0,
            last_surprise: 0.0,
        }
    }

    #[inline]
    pub fn is_trunk(&self) -> bool {
        self.role == "core" || self.role == "subcore"
    }

    /// Wave 1: LFM-2.5 Gated Short-Convolution + Continuous-Time Liquid (`CfC`) state step.
    pub fn forward_liquid_step(&mut self, v1: &[f32; MLU_DIM], v2: &[f32; MLU_DIM], v3: &[f32; MLU_DIM], dt: f32) {
        self.activation_count += 1;
        let mut next_state = [0.0f32; MLU_DIM];
        for i in 0..MLU_DIM {
            let conv = self.conv_k1[i] * v1[i] + self.conv_k2[i] * v2[i] + self.conv_k3[i] * v3[i];
            let gate = fast_sigmoid(self.gate_weights[i] * v1[i] + self.anchor_vec[i]);
            let gated_conv = fast_silu(conv) * gate;

            let tau = fast_sigmoid((self.tau_weights[i] + gated_conv.abs()) * dt);
            let target = (0.65 * self.anchor_vec[i] + 0.35 * gated_conv).tanh();
            next_state[i] = (1.0 - tau) * self.hidden_state[i] + tau * target;
        }
        normalize_in_place(&mut next_state);
        self.hidden_state.copy_from_slice(&next_state);
    }

    /// Project `hidden_state` through radial spoke `spoke_idx`.
    pub fn project_spoke(&self, spoke_idx: usize) -> [f32; MLU_DIM] {
        let mut proj = [0.0f32; MLU_DIM];
        let w = &self.spoke_weights[spoke_idx];
        let bias = self.spoke_biases[spoke_idx] * 0.1;
        for i in 0..MLU_DIM {
            proj[i] = (self.hidden_state[i] * w[i] + bias).tanh();
        }
        normalize_in_place(&mut proj);
        proj
    }

    /// Score compatibility with `target_vec` on `spoke_idx`.
    pub fn score_spoke_target(&self, spoke_idx: usize, target_name: &str, target_vec: &[f32; MLU_DIM]) -> f32 {
        let proj = self.project_spoke(spoke_idx);
        let cosine = dot_product(&proj, target_vec);
        let affinity_bonus = self.spoke_synapses[spoke_idx]
            .iter()
            .find(|s| s.target_name == target_name)
            .map(|s| s.affinity)
            .unwrap_or(0.0);
        fast_sigmoid(1.8 * cosine + affinity_bonus + self.spoke_biases[spoke_idx])
    }

    /// Online Multi-Token Prediction (MTP) sequence update in Rust.
    pub fn learn_token_sequence(&mut self, tokens: &[String], lr: f32) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }
        let eff_lr = lr.max(0.05);
        let mut total_surprise: f32 = 0.0;
        let mut pairs: usize = 0;

        for idx in 0..(tokens.len() - 1) {
            let curr = &tokens[idx];
            let next = &tokens[idx + 1];
            let row = self.token_transitions.entry(curr.clone()).or_default();
            let row_sum: f32 = row.values().sum();
            let prob = if row_sum > 0.0 {
                row.get(next).copied().unwrap_or(0.0) / row_sum
            } else {
                0.0
            };
            let surprise = 1.0 - prob;
            total_surprise += surprise;
            pairs += 1;
            *row.entry(next.clone()).or_insert(0.0) += eff_lr;

            if idx >= 1 {
                let bigram_key = format!("{} {}", tokens[idx - 1], curr);
                let b_row = self.token_transitions.entry(bigram_key).or_default();
                *b_row.entry(next.clone()).or_insert(0.0) += eff_lr * 1.25;
            }
        }
        let avg = total_surprise / (pairs.max(1) as f32);
        self.last_surprise = avg;
        self.cumulative_surprise += avg;
        avg
    }
}

/// Rust-Native 360° Spider-Web Arena holding all `MluCell` units.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpiderWebArena {
    pub cells: Vec<MluCell>,
    pub name_to_id: HashMap<String, u32>,
    pub mitosis_min_activations: u64,
    pub mitosis_min_verified_spokes: usize,
    pub total_mitosis_events: u64,
}

impl Default for SpiderWebArena {
    fn default() -> Self {
        Self::new(3, 2)
    }
}

impl SpiderWebArena {
    pub fn new(mitosis_min_activations: u64, mitosis_min_verified_spokes: usize) -> Self {
        let mut arena = Self {
            cells: Vec::with_capacity(256),
            name_to_id: HashMap::with_capacity(256),
            mitosis_min_activations,
            mitosis_min_verified_spokes,
            total_mitosis_events: 0,
        };
        arena.ensure_cell("CORE_ROOT", "core", None);
        for subcore in DEFAULT_SUBCORES {
            arena.ensure_cell(subcore, "subcore", Some("CORE_ROOT".to_string()));
        }
        arena
    }

    pub fn ensure_cell(&mut self, concept_id: &str, default_role: &str, preferred_hub: Option<String>) -> u32 {
        let clean = concept_id.trim().to_uppercase().replace(' ', "_");
        if let Some(&id) = self.name_to_id.get(&clean) {
            return id;
        }
        let hub = preferred_hub.unwrap_or_else(|| "SUBCORE_TAXONOMY".to_string());
        let id = self.cells.len() as u32;
        let cell = MluCell::new(id, &clean, default_role, if default_role == "core" { None } else { Some(hub) });
        self.cells.push(cell);
        self.name_to_id.insert(clean, id);
        id
    }

    pub fn get_anchor_array(&self, id: u32) -> [f32; MLU_DIM] {
        let mut arr = [0.0f32; MLU_DIM];
        if let Some(cell) = self.cells.get(id as usize) {
            arr.copy_from_slice(&cell.anchor_vec[..MLU_DIM]);
        }
        arr
    }

    pub fn get_hidden_array(&self, id: u32) -> [f32; MLU_DIM] {
        let mut arr = [0.0f32; MLU_DIM];
        if let Some(cell) = self.cells.get(id as usize) {
            arr.copy_from_slice(&cell.hidden_state[..MLU_DIM]);
        }
        arr
    }

    /// Wave 1 + Wave 3: Local Predictive-Coding & 3-Factor Verification Plasticity in Rust.
    pub fn learn_spoke_relation(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        learning_rate: f32,
        verified: bool,
    ) -> f32 {
        let pred_vec = deterministic_seed_vec(predicate);
        let obj_seed = deterministic_seed_vec(object);
        let spoke_idx = route_phase_spoke_idx(&pred_vec, &obj_seed);
        let default_hub = DEFAULT_SUBCORES[spoke_idx].to_string();
        let sub_id = self.ensure_cell(subject, "leaf", Some(default_hub.clone()));
        let obj_id = self.ensure_cell(object, "leaf", Some(default_hub));

        let sub_anchor = self.get_anchor_array(sub_id);
        let obj_anchor = self.get_anchor_array(obj_id);

        // Step Wave 1 (LFM Liquid State) on subject, object, and CORE_ROOT
        if let Some(sub_cell) = self.cells.get_mut(sub_id as usize) {
            sub_cell.forward_liquid_step(&obj_anchor, &pred_vec, &sub_anchor, 1.0);
        }
        if let Some(obj_cell) = self.cells.get_mut(obj_id as usize) {
            obj_cell.forward_liquid_step(&sub_anchor, &pred_vec, &obj_anchor, 1.0);
        }
        if let Some(core_cell) = self.cells.get_mut(0) {
            core_cell.forward_liquid_step(&sub_anchor, &pred_vec, &obj_anchor, 1.0);
        }

        let obj_clean = object.trim().to_uppercase().replace(' ', "_");
        let pred_clean = predicate.trim().to_lowercase();

        // Wave 3: Local Predictive-Coding update on subject cell
        let sub_cell = &mut self.cells[sub_id as usize];
        let pred_prob = sub_cell.score_spoke_target(spoke_idx, &obj_clean, &obj_anchor);
        let target_label = if verified { 1.0f32 } else { 0.0f32 };
        let surprise = (target_label - pred_prob).abs();
        sub_cell.last_surprise = surprise;
        sub_cell.cumulative_surprise += surprise;

        let eff_lr = (learning_rate * if sub_cell.is_trunk() { 0.1 } else { 1.0 }).clamp(0.005, 0.5);
        let direction = if verified { 1.0f32 } else { -1.0f32 };

        let mut new_spoke_w = [0.0f32; MLU_DIM];
        let mut new_anchor = [0.0f32; MLU_DIM];
        for i in 0..MLU_DIM {
            let grad = direction * surprise * sub_cell.hidden_state[i] * obj_anchor[i];
            new_spoke_w[i] = sub_cell.spoke_weights[spoke_idx][i] + eff_lr * grad;
            new_anchor[i] = sub_cell.anchor_vec[i] + 0.35 * eff_lr * direction * surprise * obj_anchor[i];
        }
        normalize_in_place(&mut new_spoke_w);
        normalize_in_place(&mut new_anchor);
        sub_cell.spoke_weights[spoke_idx].copy_from_slice(&new_spoke_w);
        sub_cell.anchor_vec.copy_from_slice(&new_anchor);

        let mut blended_hidden = [0.0f32; MLU_DIM];
        for i in 0..MLU_DIM {
            blended_hidden[i] = 0.8 * sub_cell.hidden_state[i] + 0.2 * new_anchor[i];
        }
        normalize_in_place(&mut blended_hidden);
        sub_cell.hidden_state.copy_from_slice(&blended_hidden);

        // Update SpokeSynapse evidence & affinity
        if let Some(syn) = sub_cell.spoke_synapses[spoke_idx]
            .iter_mut()
            .find(|s| s.target_name == obj_clean)
        {
            if verified {
                syn.pos_evidence += 1.0;
                syn.affinity = (syn.affinity + eff_lr * (0.5 + surprise)).min(3.5);
                syn.predicate = pred_clean;
            } else {
                syn.neg_evidence += 1.0;
                syn.affinity = (syn.affinity - eff_lr * (0.8 + surprise)).max(-3.5);
            }
        } else {
            sub_cell.spoke_synapses[spoke_idx].push(SpokeSynapse {
                target_id: obj_id,
                target_name: obj_clean,
                predicate: pred_clean,
                affinity: if verified {
                    (eff_lr * (0.5 + surprise)).min(3.5)
                } else {
                    (-eff_lr * (0.8 + surprise)).max(-3.5)
                },
                pos_evidence: if verified { 1.0 } else { 0.0 },
                neg_evidence: if verified { 0.0 } else { 1.0 },
            });
        }

        if verified {
            sub_cell.verified_hits += 1.0;
        } else {
            sub_cell.refuted_hits += 1.0;
        }

        // Check 5-Brake Fractal Mitosis on `sub_cell`
        let populated_spokes = sub_cell
            .spoke_synapses
            .iter()
            .filter(|edges| !edges.is_empty())
            .count();
        if sub_cell.role == "leaf"
            && !sub_cell.concept_id.starts_with("SUBCORE_")
            && sub_cell.activation_count >= self.mitosis_min_activations
            && populated_spokes >= self.mitosis_min_verified_spokes
            && sub_cell.verified_hits >= (self.mitosis_min_verified_spokes as f32)
            && sub_cell.verified_hits > sub_cell.refuted_hits
        {
            sub_cell.role = "subcore".to_string();
            self.total_mitosis_events += 1;
        }

        surprise
    }

    /// Wave 2: Predict top-k 360° spoke targets for `concept_id` on `spoke_idx` in Rust
    /// using continuous 360° angular phase coherence across the cell's radial synapses.
    pub fn predict_spoke_targets(
        &self,
        concept_id: &str,
        spoke_idx: usize,
        top_k: usize,
    ) -> Vec<SpokePredictionDto> {
        let clean = concept_id.trim().to_uppercase().replace(' ', "_");
        let Some(&cid) = self.name_to_id.get(&clean) else {
            return Vec::new();
        };
        let Some(cell) = self.cells.get(cid as usize) else {
            return Vec::new();
        };
        if spoke_idx >= NUM_SPOKES {
            return Vec::new();
        }

        let mut raw_scores: Vec<(String, f32, f32, String, usize, bool)> = Vec::new();
        let direct_non_empty = !cell.spoke_synapses[spoke_idx].is_empty();
        let spoke_range: Vec<usize> = if direct_non_empty {
            vec![spoke_idx]
        } else {
            (0..NUM_SPOKES).collect()
        };

        for s_idx in spoke_range {
            let angle_diff = ((s_idx as f32) - (spoke_idx as f32))
                * (2.0 * std::f32::consts::PI / (NUM_SPOKES as f32));
            let phase_coherence = 0.75 + 0.25 * angle_diff.cos();
            for syn in &cell.spoke_synapses[s_idx] {
                if syn.neg_evidence > syn.pos_evidence {
                    continue;
                }
                let target_vec = self.get_anchor_array(syn.target_id);
                let neural_score =
                    cell.score_spoke_target(s_idx, &syn.target_name, &target_vec) * phase_coherence;
                let total_ev = syn.pos_evidence + syn.neg_evidence;
                let nars_conf =
                    (syn.pos_evidence / total_ev.max(1.0)) * (total_ev / (total_ev + 0.5));
                raw_scores.push((
                    syn.target_name.clone(),
                    neural_score,
                    nars_conf,
                    syn.predicate.clone(),
                    s_idx,
                    syn.pos_evidence > 0.0,
                ));
            }
        }

        if raw_scores.is_empty() {
            return Vec::new();
        }

        let score_sum: f32 = raw_scores.iter().map(|x| x.1.max(1e-6)).sum();
        let norm_entropy = if raw_scores.len() <= 1 {
            0.0
        } else {
            let mut shannon = 0.0f32;
            for item in &raw_scores {
                let p = item.1.max(1e-6) / score_sum;
                shannon -= p * p.ln();
            }
            (shannon / (raw_scores.len() as f32).ln()).clamp(0.0, 1.0)
        };

        let mut results: Vec<SpokePredictionDto> = raw_scores
            .into_iter()
            .map(|(target_name, neural_score, nars_conf, pred_label, s_idx, is_ver)| {
                let combined = 0.65 * nars_conf + 0.35 * neural_score;
                SpokePredictionDto {
                    source_concept: cell.concept_id.clone(),
                    spoke: SPOKE_NAMES[s_idx].to_string(),
                    predicate: pred_label,
                    target_concept: target_name,
                    neural_score,
                    nars_confidence: nars_conf,
                    epistemic_entropy: norm_entropy,
                    combined_confidence: combined,
                    verified: is_ver,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.combined_confidence
                .partial_cmp(&a.combined_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Phase 11: Collect all reachable positive taxonomic ancestors (`is_a`) of `start_concept`.
    pub fn collect_positive_ancestors(&self, start_concept: &str) -> std::collections::HashSet<String> {
        let clean = start_concept.trim().to_uppercase().replace(' ', "_");
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        visited.insert(clean.clone());
        queue.push_back(clean);

        while let Some(curr) = queue.pop_front() {
            let Some(&cid) = self.name_to_id.get(&curr) else {
                continue;
            };
            let Some(cell) = self.cells.get(cid as usize) else {
                continue;
            };
            for edges in &cell.spoke_synapses {
                for syn in edges {
                    if syn.pos_evidence > syn.neg_evidence
                        && syn.predicate == "is_a"
                        && visited.insert(syn.target_name.clone())
                    {
                        queue.push_back(syn.target_name.clone());
                    }
                }
            }
        }
        visited
    }

    /// Phase 11: Collect all mutually exclusive concepts linked to any member of `concept_set`.
    pub fn collect_disjoint_peers(
        &self,
        concept_set: &std::collections::HashSet<String>,
    ) -> std::collections::HashSet<String> {
        let mut disjoints = std::collections::HashSet::new();
        for cell in &self.cells {
            let sub_in = concept_set.contains(&cell.concept_id);
            for edges in &cell.spoke_synapses {
                for syn in edges {
                    let is_disjoint_edge = (syn.predicate == "disjoint_with"
                        && syn.pos_evidence > syn.neg_evidence)
                        || (syn.neg_evidence > syn.pos_evidence);
                    if !is_disjoint_edge {
                        continue;
                    }
                    if sub_in {
                        disjoints.insert(syn.target_name.clone());
                    } else if concept_set.contains(&syn.target_name) {
                        disjoints.insert(cell.concept_id.clone());
                    }
                }
            }
        }
        disjoints
    }

    /// Phase 11: Native Rust Invariant Gate Verifier (`I_DAG` Acyclicity + `I_MUTEX` Mutual Exclusion).
    pub fn verify_spoke_invariants(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> (bool, &'static str) {
        let sub_clean = subject.trim().to_uppercase().replace(' ', "_");
        let obj_clean = object.trim().to_uppercase().replace(' ', "_");
        let pred_clean = predicate.trim().to_lowercase();

        if sub_clean.is_empty() || obj_clean.is_empty() || sub_clean == obj_clean {
            return (false, "I_DAG");
        }
        if pred_clean == "disjoint_with" {
            let sub_anc = self.collect_positive_ancestors(&sub_clean);
            let obj_anc = self.collect_positive_ancestors(&obj_clean);
            if sub_anc.contains(&obj_clean) || obj_anc.contains(&sub_clean) {
                return (false, "I_MUTEX");
            }
            return (true, "PASSED");
        }

        // Gate 1: I_DAG (check if object already reaches subject)
        let obj_anc = self.collect_positive_ancestors(&obj_clean);
        if obj_anc.contains(&sub_clean) {
            return (false, "I_DAG");
        }

        // Gate 2: I_MUTEX (check if subject's ancestor closure is disjoint with object's ancestor closure)
        let sub_anc = self.collect_positive_ancestors(&sub_clean);
        let sub_disjoints = self.collect_disjoint_peers(&sub_anc);
        if !sub_disjoints.is_disjoint(&obj_anc) {
            return (false, "I_MUTEX");
        }

        (true, "PASSED")
    }

    /// Pure-Rust 360° Liquid-Resonance Stream Learner (zero hardcoded word lists or regexes).
    ///
    /// Uses Shannon character-entropy salience `token_information_salience(w)` to separate
    /// high-information concept anchors from low-entropy structural operators, verifies every
    /// candidate link via native `verify_spoke_invariants` (`I_DAG` & `I_MUTEX`), updates local
    /// JEPA spoke weights + Multi-Token Prediction (MTP) transitions, and triggers 5-Brake Fractal Mitosis.
    pub fn train_raw_stream(&mut self, text: &str, learning_rate: f32) -> StreamTrainStatsDto {
        let mut sentences_processed: usize = 0;
        let mut tokens_processed: usize = 0;
        let mut spoke_updates: usize = 0;
        let mut total_surprise: f32 = 0.0;
        let mut surprise_count: usize = 0;
        let initial_mitosis = self.total_mitosis_events;
        let mut active_discourse_focus: Option<String> = None;

        for raw_sent in text.split(['.', '!', '?', '\n']) {
            let trimmed = raw_sent.trim();
            if trimmed.is_empty() {
                continue;
            }
            let tokens: Vec<String> = trimmed
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_lowercase())
                .collect();

            if tokens.len() < 2 {
                continue;
            }
            sentences_processed += 1;
            tokens_processed += tokens.len();

            // 1. Update CORE_ROOT Multi-Token Prediction (MTP)
            if let Some(core) = self.cells.get_mut(0) {
                let s = core.learn_token_sequence(&tokens, learning_rate);
                total_surprise += s;
                surprise_count += 1;
            }

            // 2. Slide 3-tap LFM-2.5 liquid window over high-salience concept anchors (Shannon H(w) * ln(1+|w|) >= 2.25)
            let anchor_indices: Vec<usize> = tokens
                .iter()
                .enumerate()
                .filter(|(_, w)| token_information_salience(w) >= 2.25)
                .map(|(i, _)| i)
                .collect();

            // Phase 29: Liquid Coreference Binding for low-entropy leading anaphors
            if let Some(&first_idx) = anchor_indices.first() {
                if first_idx == 0
                    || (first_idx == 1 && token_information_salience(&tokens[0]) < 1.85)
                {
                    if first_idx == 1 && active_discourse_focus.is_some() && anchor_indices.len() == 1 {
                        if let Some(ref focus) = active_discourse_focus {
                            let target_tok = &tokens[first_idx];
                            if focus != target_tok {
                                let pred = &tokens[0];
                                let (passed, _) =
                                    self.verify_spoke_invariants(focus, pred, target_tok);
                                let s = self.learn_spoke_relation(
                                    focus,
                                    pred,
                                    target_tok,
                                    learning_rate * 0.5,
                                    passed,
                                );
                                total_surprise += s;
                                surprise_count += 1;
                                spoke_updates += 1;
                            }
                        }
                    } else {
                        active_discourse_focus = Some(tokens[first_idx].clone());
                    }
                } else if first_idx >= 2 {
                    if let Some(ref focus) = active_discourse_focus {
                        let target_tok = &tokens[first_idx];
                        if focus != target_tok {
                            let pred = tokens[1..first_idx].join("_");
                            let (passed, _) =
                                self.verify_spoke_invariants(focus, &pred, target_tok);
                            let s = self.learn_spoke_relation(
                                focus,
                                &pred,
                                target_tok,
                                learning_rate * 0.5,
                                passed,
                            );
                            total_surprise += s;
                            surprise_count += 1;
                            spoke_updates += 1;
                        }
                    }
                }
            }

            for idx in 0..anchor_indices.len() {
                let tok_idx = anchor_indices[idx];
                let curr_tok = &tokens[tok_idx];
                let prev1 = if idx >= 1 {
                    &tokens[anchor_indices[idx - 1]]
                } else {
                    curr_tok
                };
                let prev2 = if idx >= 2 {
                    &tokens[anchor_indices[idx - 2]]
                } else {
                    prev1
                };

                let cid = self.ensure_cell(curr_tok, "leaf", None);
                let v1 = deterministic_seed_vec(curr_tok);
                let v2 = deterministic_seed_vec(prev1);
                let v3 = deterministic_seed_vec(prev2);

                if let Some(cell) = self.cells.get_mut(cid as usize) {
                    cell.forward_liquid_step(&v1, &v2, &v3, 1.0);
                    cell.learn_token_sequence(&tokens, learning_rate * 0.8);
                }

                if idx + 1 < anchor_indices.len() {
                    let next_tok_idx = anchor_indices[idx + 1];
                    let next_tok = &tokens[next_tok_idx];
                    if next_tok != curr_tok {
                        let intervening_pred = if next_tok_idx > tok_idx + 1 {
                            tokens[tok_idx + 1..next_tok_idx].join("_")
                        } else {
                            "co_occurs_with".to_string()
                        };
                        let (passed, _gate) =
                            self.verify_spoke_invariants(curr_tok, &intervening_pred, next_tok);
                        let s = self.learn_spoke_relation(
                            curr_tok,
                            &intervening_pred,
                            next_tok,
                            learning_rate * 0.5,
                            passed,
                        );
                        total_surprise += s;
                        surprise_count += 1;
                        spoke_updates += 1;
                    }
                }
            }
        }

        let avg_surprise = if surprise_count > 0 {
            total_surprise / (surprise_count as f32)
        } else {
            0.0
        };

        StreamTrainStatsDto {
            sentences_processed,
            tokens_processed,
            spoke_updates,
            average_surprise: avg_surprise,
            total_cells: self.cells.len(),
            mitosis_promotions: (self.total_mitosis_events - initial_mitosis) as usize,
        }
    }

    /// Generate a fluent multi-token continuation from the Rust 360° Spider-Web Arena.
    pub fn generate_from_prompt(&self, prompt: &str, max_tokens: usize) -> String {
        let seed_tokens: Vec<String> = prompt
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect();

        if seed_tokens.is_empty() || max_tokens == 0 {
            return String::new();
        }

        // Locate focal cell if any seed token matches an active concept cell
        let mut focal_cell_id: usize = 0;
        for tok in seed_tokens.iter().rev() {
            let clean = tok.to_uppercase();
            if let Some(&id) = self.name_to_id.get(&clean) {
                focal_cell_id = id as usize;
                break;
            }
        }

        let mut history = seed_tokens;
        let mut generated: Vec<String> = Vec::with_capacity(max_tokens);

        for _ in 0..max_tokens {
            let mut best_next: Option<String> = None;
            let mut best_weight: f32 = 0.0;

            let check_ids = if focal_cell_id != 0 {
                vec![focal_cell_id, 0usize]
            } else {
                vec![0usize]
            };

            for &cid in &check_ids {
                let Some(cell) = self.cells.get(cid) else {
                    continue;
                };
                if history.len() >= 2 {
                    let bigram = format!(
                        "{} {}",
                        history[history.len() - 2],
                        history[history.len() - 1]
                    );
                    if let Some(row) = cell.token_transitions.get(&bigram) {
                        for (cand, &w) in row {
                            if !generated.contains(cand) && w * 1.35 > best_weight {
                                best_weight = w * 1.35;
                                best_next = Some(cand.clone());
                            }
                        }
                    }
                }
                let last_tok = &history[history.len() - 1];
                if let Some(row) = cell.token_transitions.get(last_tok) {
                    for (cand, &w) in row {
                        if !generated.contains(cand) && w > best_weight {
                            best_weight = w;
                            best_next = Some(cand.clone());
                        }
                    }
                }
            }

            if let Some(chosen) = best_next {
                history.push(chosen.clone());
                generated.push(chosen);
            } else {
                break;
            }
        }

        generated.join(" ")
    }

    /// Persist the entire 360° Spider-Web Arena state to a snapshot file on disk.
    pub fn save_snapshot_file(&self, path: &str) -> bool {
        let Ok(payload) = serde_json::to_vec(self) else {
            return false;
        };
        std::fs::write(path, payload).is_ok()
    }

    /// Restore the 360° Spider-Web Arena state from a snapshot file on disk.
    pub fn load_snapshot_file(&mut self, path: &str) -> bool {
        let Ok(bytes) = std::fs::read(path) else {
            return false;
        };
        let Ok(restored) = serde_json::from_slice::<SpiderWebArena>(&bytes) else {
            return false;
        };
        *self = restored;
        true
    }

    /// Serialize the entire 360° Spider-Web Arena state to a JSON string (for SQLite blob/text storage).
    pub fn dump_snapshot_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Restore the 360° Spider-Web Arena state from a JSON string.
    pub fn restore_snapshot_json(&mut self, payload: &str) -> bool {
        let Ok(restored) = serde_json::from_str::<SpiderWebArena>(payload) else {
            return false;
        };
        *self = restored;
        true
    }

    /// Phase 1: Native Rust 360° Multi-Hop Beam Traversal & Neural Continuation (`query_360_beam`).
    ///
    /// Executes Wave 1 (liquid state step), Wave 2 (direct 6-spoke predictions + multi-hop
    /// confidence-guided beam expansion with 0.88^hop decay), deduplication, and Multi-Token
    /// Prediction continuation in a single native Rust pass.
    pub fn query_360_beam(
        &mut self,
        concept_id: &str,
        max_hops: usize,
        top_k: usize,
    ) -> Query360BeamDto {
        let clean = concept_id.trim().to_uppercase().replace(' ', "_");
        let Some(&cid) = self.name_to_id.get(&clean) else {
            return Query360BeamDto {
                focal_concept: clean,
                active_subcore: "CORE_ROOT".to_string(),
                role: "unknown".to_string(),
                activated_cells: vec!["CORE_ROOT".to_string()],
                predictions: Vec::new(),
                generated_continuation: String::new(),
                fluent_synthesis: String::new(),
            };
        };

        let core_hidden = self.get_hidden_array(0);
        if let Some(focal_cell) = self.cells.get_mut(cid as usize) {
            focal_cell.forward_liquid_step(&core_hidden, &core_hidden, &core_hidden, 1.0);
        }

        let (role, active_subcore) = {
            let cell = &self.cells[cid as usize];
            (
                cell.role.clone(),
                cell.parent_hub
                    .clone()
                    .unwrap_or_else(|| "CORE_ROOT".to_string()),
            )
        };

        let mut activated_cells = vec!["CORE_ROOT".to_string()];
        if active_subcore != "CORE_ROOT" {
            activated_cells.push(active_subcore.clone());
        }
        activated_cells.push(clean.clone());

        let mut all_preds: Vec<SpokePredictionDto> = Vec::new();
        for s_idx in 0..NUM_SPOKES {
            if !self.cells[cid as usize].spoke_synapses[s_idx].is_empty() {
                all_preds.extend(self.predict_spoke_targets(&clean, s_idx, 4));
            }
        }

        // Multi-hop confidence-guided beam expansion across the 360° Spider-Web
        let mut frontier: Vec<String> = all_preds
            .iter()
            .filter(|p| p.verified && p.predicate != "disjoint_with" && p.nars_confidence >= 0.45)
            .map(|p| p.target_concept.clone())
            .collect();

        let mut visited = std::collections::HashSet::new();
        visited.insert(clean.clone());
        let mut hop: usize = 1;
        let decay_base: f32 = 0.88;

        while !frontier.is_empty() && hop < max_hops.max(1) {
            let factor = decay_base.powi(hop as i32);
            let mut next_frontier: Vec<String> = Vec::new();

            for parent_cid in frontier {
                if !visited.insert(parent_cid.clone()) {
                    continue;
                }
                let Some(&p_id) = self.name_to_id.get(&parent_cid) else {
                    continue;
                };
                if !activated_cells.contains(&parent_cid) {
                    activated_cells.push(parent_cid.clone());
                }
                for s_idx in 0..NUM_SPOKES {
                    if self.cells[p_id as usize].spoke_synapses[s_idx].is_empty() {
                        continue;
                    }
                    let parent_preds = self.predict_spoke_targets(&parent_cid, s_idx, 3);
                    for pp in parent_preds {
                        if pp.target_concept == clean
                            || !pp.verified
                            || pp.predicate == "disjoint_with"
                        {
                            continue;
                        }
                        if pp.nars_confidence >= 0.45 {
                            next_frontier.push(pp.target_concept.clone());
                        }
                        all_preds.push(SpokePredictionDto {
                            source_concept: clean.clone(),
                            spoke: pp.spoke,
                            predicate: pp.predicate,
                            target_concept: pp.target_concept,
                            neural_score: (pp.neural_score * factor * 10000.0).round() / 10000.0,
                            nars_confidence: (pp.nars_confidence * factor * 10000.0).round()
                                / 10000.0,
                            epistemic_entropy: pp.epistemic_entropy,
                            combined_confidence: (pp.combined_confidence * factor * 10000.0)
                                .round()
                                / 10000.0,
                            verified: pp.verified,
                        });
                    }
                }
            }
            frontier = next_frontier;
            hop += 1;
        }

        // Deduplicate by (spoke, predicate, target_concept) keeping highest combined_confidence
        let mut dedup: HashMap<(String, String, String), SpokePredictionDto> = HashMap::new();
        for pred in all_preds {
            let key = (
                pred.spoke.clone(),
                pred.predicate.clone(),
                pred.target_concept.clone(),
            );
            let replace = match dedup.get(&key) {
                Some(existing) => pred.combined_confidence > existing.combined_confidence,
                None => true,
            };
            if replace {
                dedup.insert(key, pred);
            }
        }

        let mut sorted: Vec<SpokePredictionDto> = dedup.into_values().collect();
        sorted.sort_by(|a, b| {
            b.combined_confidence
                .partial_cmp(&a.combined_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(top_k.max(1));

        let prompt = clean.to_lowercase().replace('_', " ");
        let generated_continuation = self.generate_from_prompt(&prompt, 8);
        let fluent_synthesis =
            self.synthesize_360_wave(&clean, &sorted, &generated_continuation);

        Query360BeamDto {
            focal_concept: clean,
            active_subcore,
            role,
            activated_cells,
            predictions: sorted,
            generated_continuation,
            fluent_synthesis,
        }
    }

    /// Phase 8: Native Rust 360° Wave-to-Token Sequence Decoder (`synthesize_360_wave`).
    ///
    /// Decodes active 360° spoke predictions and local liquid token transitions into a
    /// coherent natural-language realization without hardcoded templates.
    pub fn synthesize_360_wave(
        &self,
        focal_concept: &str,
        predictions: &[SpokePredictionDto],
        continuation: &str,
    ) -> String {
        let subj = focal_concept.to_lowercase().replace('_', " ");
        let mut clauses: Vec<String> = Vec::new();
        let mut seen_pred_groups: Vec<(String, Vec<String>)> = Vec::new();

        for pred in predictions.iter().take(6) {
            if !pred.verified || pred.predicate == "co_occurs_with" {
                continue;
            }
            let verb = pred.predicate.replace('_', " ");
            let obj = pred.target_concept.to_lowercase().replace('_', " ");
            if let Some((_, targets)) = seen_pred_groups.iter_mut().find(|(v, _)| v == &verb) {
                if !targets.contains(&obj) {
                    targets.push(obj);
                }
            } else {
                seen_pred_groups.push((verb, vec![obj]));
            }
        }

        for (verb, objs) in seen_pred_groups {
            let joined = match objs.len() {
                0 => continue,
                1 => objs[0].clone(),
                2 => format!("{} and {}", objs[0], objs[1]),
                _ => {
                    let head = objs[..objs.len() - 1].join(", ");
                    format!("{}, and {}", head, objs[objs.len() - 1])
                }
            };
            clauses.push(format!("{} {}", verb, joined));
        }

        if clauses.is_empty() {
            if !continuation.trim().is_empty() {
                let mut text = format!("{} {}", subj, continuation.trim());
                if let Some(first) = text.get_mut(0..1) {
                    first.make_ascii_uppercase();
                }
                if !text.ends_with('.') {
                    text.push('.');
                }
                return text;
            }
            return String::new();
        }

        let mut head_subj = subj;
        if let Some(first) = head_subj.get_mut(0..1) {
            first.make_ascii_uppercase();
        }
        format!("{} {}.", head_subj, clauses.join("; "))
    }

    /// Phase 2: Native Rust Sleep-Cycle Consolidation (`consolidate_arena`).
    ///
    /// 1. Replays verified spoke synapses (`pos_evidence > neg_evidence`) across all cells to
    ///    sharpen 360° radial spoke separation and reduce prediction surprise.
    /// 2. Prunes low-weight noise transitions (`w < min_transition_weight`) from `token_transitions`
    ///    so memory stays bounded over long continuous streams.
    pub fn consolidate_arena(
        &mut self,
        min_transition_weight: f32,
        replay_lr: f32,
    ) -> ConsolidationReportDto {
        let mut replayed_edges: usize = 0;
        let mut total_surprise: f32 = 0.0;
        let mut pruned_transitions: usize = 0;
        let initial_mitosis = self.total_mitosis_events;

        // Collect verified edges to replay
        let mut replay_queue: Vec<(String, String, String)> = Vec::new();
        for cell in &mut self.cells {
            // Prune weak token transitions
            let mut empty_keys: Vec<String> = Vec::new();
            for (ctx_key, row) in cell.token_transitions.iter_mut() {
                let before = row.len();
                row.retain(|_, &mut w| w >= min_transition_weight);
                pruned_transitions += before.saturating_sub(row.len());
                if row.is_empty() {
                    empty_keys.push(ctx_key.clone());
                }
            }
            for k in empty_keys {
                cell.token_transitions.remove(&k);
            }

            for spoke_edges in &cell.spoke_synapses {
                for syn in spoke_edges {
                    if syn.pos_evidence > syn.neg_evidence {
                        replay_queue.push((
                            cell.concept_id.clone(),
                            syn.predicate.clone(),
                            syn.target_name.clone(),
                        ));
                    }
                }
            }
        }

        let eff_lr = replay_lr.clamp(0.02, 0.35);
        for (subj, pred, obj) in replay_queue {
            let s = self.learn_spoke_relation(&subj, &pred, &obj, eff_lr, true);
            total_surprise += s;
            replayed_edges += 1;
        }

        let avg_surprise = if replayed_edges > 0 {
            total_surprise / (replayed_edges as f32)
        } else {
            0.0
        };

        ConsolidationReportDto {
            replayed_edges,
            pruned_transitions,
            average_replay_surprise: avg_surprise,
            mitosis_promotions: (self.total_mitosis_events - initial_mitosis) as usize,
            total_cells: self.cells.len(),
        }
    }

    /// Phase 14: Native Rust 360° Bidirectional Wave Intersection (`compare_concepts_360`).
    ///
    /// Launches simultaneous 360° multi-hop waves from `concept_a` and `concept_b`, finding
    /// shared 360° attractors (convergence), distinct phase spokes (divergence), `I_MUTEX`
    /// mutual exclusion boundaries, and synthesizing a grounded comparative explanation.
    pub fn compare_concepts_360(
        &mut self,
        concept_a: &str,
        concept_b: &str,
        max_hops: usize,
    ) -> Compare360Dto {
        let beam_a = self.query_360_beam(concept_a, max_hops, 16);
        let beam_b = self.query_360_beam(concept_b, max_hops, 16);
        let clean_a = beam_a.focal_concept.clone();
        let clean_b = beam_b.focal_concept.clone();

        let vec_a = self
            .name_to_id
            .get(&clean_a)
            .map(|&id| self.get_anchor_array(id))
            .unwrap_or_else(|| deterministic_seed_vec(&clean_a));
        let vec_b = self
            .name_to_id
            .get(&clean_b)
            .map(|&id| self.get_anchor_array(id))
            .unwrap_or_else(|| deterministic_seed_vec(&clean_b));
        let anchor_similarity = ((dot_product(&vec_a, &vec_b) * 10000.0).round()) / 10000.0;

        let (passed_ab, gate_ab) = self.verify_spoke_invariants(&clean_a, "is_a", &clean_b);
        let (passed_ba, gate_ba) = self.verify_spoke_invariants(&clean_b, "is_a", &clean_a);
        let mutually_exclusive =
            (!passed_ab && gate_ab == "I_MUTEX") || (!passed_ba && gate_ba == "I_MUTEX");

        let mut shared_attractors: Vec<SharedAttractorDto> = Vec::new();
        let mut shared_targets_set = std::collections::HashSet::new();

        for pa in &beam_a.predictions {
            if !pa.verified || pa.predicate == "co_occurs_with" || pa.predicate == "disjoint_with" {
                continue;
            }
            if let Some(pb) = beam_b.predictions.iter().find(|pb| {
                pb.verified
                    && pb.target_concept == pa.target_concept
                    && pb.predicate != "co_occurs_with"
                    && pb.predicate != "disjoint_with"
            }) {
                if shared_targets_set.insert(pa.target_concept.clone()) {
                    shared_attractors.push(SharedAttractorDto {
                        predicate: pa.predicate.clone(),
                        target_concept: pa.target_concept.clone(),
                        spoke: pa.spoke.clone(),
                        combined_confidence: ((0.5
                            * (pa.combined_confidence + pb.combined_confidence)
                            * 10000.0)
                            .round())
                            / 10000.0,
                    });
                }
            }
        }

        let distinct_a: Vec<SpokePredictionDto> = beam_a
            .predictions
            .into_iter()
            .filter(|p| {
                p.verified
                    && p.predicate != "co_occurs_with"
                    && p.predicate != "disjoint_with"
                    && !shared_targets_set.contains(&p.target_concept)
            })
            .take(5)
            .collect();

        let distinct_b: Vec<SpokePredictionDto> = beam_b
            .predictions
            .into_iter()
            .filter(|p| {
                p.verified
                    && p.predicate != "co_occurs_with"
                    && p.predicate != "disjoint_with"
                    && !shared_targets_set.contains(&p.target_concept)
            })
            .take(5)
            .collect();

        let name_a = clean_a.to_lowercase().replace('_', " ");
        let name_b = clean_b.to_lowercase().replace('_', " ");

        let shared_clause = if shared_attractors.is_empty() {
            format!(
                "{} and {} have no verified shared 360° attractors yet",
                name_a, name_b
            )
        } else {
            let parts: Vec<String> = shared_attractors
                .iter()
                .take(3)
                .map(|s| {
                    format!(
                        "{} {}",
                        s.predicate.replace('_', " "),
                        s.target_concept.to_lowercase().replace('_', " ")
                    )
                })
                .collect();
            format!("Both {} and {} {}", name_a, name_b, parts.join(" and "))
        };

        let diff_a: Vec<String> = distinct_a
            .iter()
            .take(2)
            .map(|p| {
                format!(
                    "{} {}",
                    p.predicate.replace('_', " "),
                    p.target_concept.to_lowercase().replace('_', " ")
                )
            })
            .collect();
        let diff_b: Vec<String> = distinct_b
            .iter()
            .take(2)
            .map(|p| {
                format!(
                    "{} {}",
                    p.predicate.replace('_', " "),
                    p.target_concept.to_lowercase().replace('_', " ")
                )
            })
            .collect();

        let mut synthesis = shared_clause;
        if !diff_a.is_empty() || !diff_b.is_empty() {
            let a_str = if diff_a.is_empty() {
                String::new()
            } else {
                format!("{} {}", name_a, diff_a.join(" and "))
            };
            let b_str = if diff_b.is_empty() {
                String::new()
            } else {
                format!("{} {}", name_b, diff_b.join(" and "))
            };
            if !a_str.is_empty() && !b_str.is_empty() {
                synthesis.push_str(&format!("; however, {}, whereas {}", a_str, b_str));
            } else if !a_str.is_empty() {
                synthesis.push_str(&format!("; specifically, {}", a_str));
            } else if !b_str.is_empty() {
                synthesis.push_str(&format!("; specifically, {}", b_str));
            }
        }
        if mutually_exclusive {
            synthesis.push_str(" (mutually exclusive via I_MUTEX)");
        }
        synthesis.push('.');
        if let Some(first) = synthesis.get_mut(0..1) {
            first.make_ascii_uppercase();
        }

        Compare360Dto {
            concept_a: clean_a,
            concept_b: clean_b,
            anchor_similarity,
            mutually_exclusive,
            shared_attractors,
            distinct_a,
            distinct_b,
            comparative_synthesis: synthesis,
        }
    }

    /// Phase 15: Native Rust Analogical Hypothesis Induction & Epistemic Curiosity (`discover_epistemic_hypotheses`).
    ///
    /// 1. Discovers invariant-safe (`I_DAG` & `I_MUTEX` verified) analogical spoke hypotheses
    ///    between concepts that share taxonomic/functional attractors.
    /// 2. Ranks active concepts with high 360° radial spoke sparsity as Epistemic Curiosity Gaps.
    pub fn discover_epistemic_hypotheses(&self, max_items: usize) -> CuriosityReportDto {
        let limit = max_items.max(1);
        let mut curiosity_gaps: Vec<CuriosityGapDto> = Vec::new();

        // Collect concept cells (including promoted domain subcores) and their positive spoke edges
        let mut concept_edges: Vec<(usize, String, Vec<(String, String, usize)>)> = Vec::new();
        let mut target_in_degree: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for (idx, cell) in self.cells.iter().enumerate() {
            if cell.concept_id == "CORE_ROOT" || cell.concept_id.starts_with("SUBCORE_") {
                continue;
            }
            let mut pos_edges: Vec<(String, String, usize)> = Vec::new();
            let mut populated_spokes: usize = 0;
            for (s_idx, edges) in cell.spoke_synapses.iter().enumerate() {
                let mut spoke_has_pos = false;
                for syn in edges {
                    if syn.pos_evidence > syn.neg_evidence
                        && syn.predicate != "co_occurs_with"
                        && syn.predicate != "disjoint_with"
                        && (!self.name_to_id.contains_key(&syn.predicate.to_uppercase())
                            || syn.predicate.contains('_'))
                    {
                        *target_in_degree.entry(syn.target_name.clone()).or_insert(0) += 1;
                        pos_edges.push((syn.predicate.clone(), syn.target_name.clone(), s_idx));
                        spoke_has_pos = true;
                    }
                }
                if spoke_has_pos {
                    populated_spokes += 1;
                }
            }

            let spoke_sparsity =
                (((NUM_SPOKES.saturating_sub(populated_spokes)) as f32) / (NUM_SPOKES as f32)
                    * 10000.0)
                    .round()
                    / 10000.0;
            if spoke_sparsity > 0.30 {
                let missing_spokes: Vec<String> = (0..NUM_SPOKES)
                    .filter(|&s_idx| {
                        !cell.spoke_synapses[s_idx]
                            .iter()
                            .any(|syn| syn.pos_evidence > syn.neg_evidence)
                    })
                    .map(|s_idx| SPOKE_NAMES[s_idx].to_string())
                    .collect();
                curiosity_gaps.push(CuriosityGapDto {
                    concept_id: cell.concept_id.clone(),
                    populated_spokes,
                    spoke_sparsity,
                    missing_spokes,
                });
            }

            if !pos_edges.is_empty() {
                concept_edges.push((idx, cell.concept_id.clone(), pos_edges));
            }
        }

        curiosity_gaps.sort_by(|a, b| {
            b.spoke_sparsity
                .partial_cmp(&a.spoke_sparsity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        curiosity_gaps.truncate(limit);

        // Induce invariant-safe analogical hypotheses across pairs sharing an attractor
        let mut hypotheses: Vec<AnalogicalHypothesisDto> = Vec::new();
        let mut seen_hyp: std::collections::HashSet<(String, String, String)> =
            std::collections::HashSet::new();

        for i in 0..concept_edges.len() {
            for j in 0..concept_edges.len() {
                if i == j {
                    continue;
                }
                let (idx_a, ref cid_a, ref edges_a) = concept_edges[i];
                let (idx_b, ref cid_b, ref edges_b) = concept_edges[j];

                // Check if cid_a and cid_b are mutually exclusive first
                let (compat_ab, gate_ab) = self.verify_spoke_invariants(cid_a, "is_a", cid_b);
                if !compat_ab && gate_ab == "I_MUTEX" {
                    continue;
                }

                // Find the most specific shared positive target between A and B (lowest in-degree)
                let targets_b: std::collections::HashMap<&str, &str> = edges_b
                    .iter()
                    .map(|(p, t, _)| (t.as_str(), p.as_str()))
                    .collect();
                let Some((pred_shared_a, shared_target, _)) = edges_a
                    .iter()
                    .filter(|(_, t, _)| targets_b.contains_key(t.as_str()))
                    .min_by_key(|(_, t, _)| *target_in_degree.get(t).unwrap_or(&999))
                else {
                    continue;
                };

                let vec_a = self.get_anchor_array(idx_a as u32);
                let vec_b = self.get_anchor_array(idx_b as u32);
                let sim = dot_product(&vec_a, &vec_b).clamp(-1.0, 1.0);
                let in_deg = (*target_in_degree.get(shared_target).unwrap_or(&2)).max(1) as f32;
                let specificity_bonus = (1.0 / in_deg).clamp(0.05, 0.50);
                let same_pred_bonus = if targets_b.get(shared_target.as_str()) == Some(&pred_shared_a.as_str()) {
                    0.15
                } else {
                    0.0
                };

                // Propose non-taxonomic properties/capabilities of A for B if invariant-safe
                for (pred_a, target_a, s_idx) in edges_a {
                    if pred_a == "is_a"
                        || target_a == cid_b
                        || targets_b.contains_key(target_a.as_str())
                    {
                        continue;
                    }
                    let (passed, _gate) = self.verify_spoke_invariants(cid_b, pred_a, target_a);
                    if !passed {
                        continue;
                    }
                    let key = (cid_b.clone(), pred_a.clone(), target_a.clone());
                    if seen_hyp.insert(key) {
                        let conf = ((0.45 + 0.25 * sim.max(0.0) + specificity_bonus + same_pred_bonus)
                            .clamp(0.1, 0.99)
                            * 10000.0)
                            .round()
                            / 10000.0;
                        hypotheses.push(AnalogicalHypothesisDto {
                            subject_concept: cid_b.clone(),
                            predicate: pred_a.clone(),
                            target_concept: target_a.clone(),
                            spoke: SPOKE_NAMES[*s_idx].to_string(),
                            analog_source: cid_a.clone(),
                            shared_attractor: shared_target.clone(),
                            confidence: conf,
                        });
                    }
                }
            }
        }

        hypotheses.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hypotheses.truncate(limit);

        CuriosityReportDto {
            analogical_hypotheses: hypotheses,
            curiosity_gaps,
        }
    }

    /// Phase 17: Native Rust Causal & Counterfactual Wave Simulation (`simulate_causal_wave_360`).
    ///
    /// Propagates a directed wave from `root_concept` up to `max_hops`, weighting causal,
    /// procedural, and dynamical transitions (`spokes 2, 4, 6`) while tracking multi-hop
    /// downstream impact chains (`affected_concepts`, `causal_steps`, `causal_synthesis`).
    pub fn simulate_causal_wave_360(&mut self, root_concept: &str, max_hops: usize) -> CausalSimulationDto {
        let clean_root = root_concept.trim().to_uppercase().replace(' ', "_");
        let hops = max_hops.clamp(1, 5);
        let Some(&start_id) = self.name_to_id.get(&clean_root) else {
            return CausalSimulationDto {
                root_concept: clean_root.clone(),
                affected_concepts: Vec::new(),
                causal_steps: Vec::new(),
                causal_synthesis: format!(
                    "I do not yet have verified causal or procedural links starting from '{}'.",
                    clean_root.to_lowercase().replace('_', " ")
                ),
            };
        };

        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(clean_root.clone());
        let mut frontier: Vec<(u32, String, f32)> = vec![(start_id, clean_root.clone(), 1.0)];
        let mut causal_steps: Vec<CausalStepDto> = Vec::new();
        let mut affected_set: Vec<String> = Vec::new();

        for hop in 1..=hops {
            let mut next_frontier: Vec<(u32, String, f32)> = Vec::new();
            for (cid, cname, parent_conf) in &frontier {
                let Some(cell) = self.cells.get(*cid as usize) else {
                    continue;
                };
                for (s_idx, edges) in cell.spoke_synapses.iter().enumerate() {
                    // Harmonic spoke weighting: prioritize causality (2), procedural (4), dynamics (6), and spatial/taxonomy bridges
                    let spoke_bonus: f32 = if s_idx == 2 || s_idx == 4 || s_idx == 6 {
                        1.0
                    } else {
                        0.75
                    };
                    for syn in edges {
                        if syn.pos_evidence <= syn.neg_evidence
                            || syn.predicate == "co_occurs_with"
                            || syn.predicate == "disjoint_with"
                        {
                            continue;
                        }
                        let total_ev = syn.pos_evidence + syn.neg_evidence;
                        let nars_conf = total_ev / (total_ev + 1.0);
                        let step_conf =
                            (parent_conf * nars_conf * spoke_bonus * (0.90f32).powi((hop - 1) as i32))
                                .clamp(0.05, 0.99);
                        let rounded_conf = (step_conf * 10000.0).round() / 10000.0;

                        causal_steps.push(CausalStepDto {
                            hop,
                            source_concept: cname.clone(),
                            predicate: syn.predicate.clone(),
                            target_concept: syn.target_name.clone(),
                            spoke: SPOKE_NAMES[s_idx].to_string(),
                            confidence: rounded_conf,
                        });

                        if !affected_set.contains(&syn.target_name) && syn.target_name != clean_root {
                            affected_set.push(syn.target_name.clone());
                        }

                        if visited.insert(syn.target_name.clone()) {
                            if let Some(&tid) = self.name_to_id.get(&syn.target_name) {
                                next_frontier.push((tid, syn.target_name.clone(), rounded_conf));
                            }
                        }
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        causal_steps.sort_by(|a, b| {
            a.hop
                .cmp(&b.hop)
                .then_with(|| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
        });

        let root_low = clean_root.to_lowercase().replace('_', " ");
        let causal_synthesis = if causal_steps.is_empty() {
            format!(
                "No downstream causal or procedural chain is currently known for {}.",
                root_low
            )
        } else {
            let clauses: Vec<String> = causal_steps
                .iter()
                .take(4)
                .map(|st| {
                    let s_low = st.source_concept.to_lowercase().replace('_', " ");
                    let p_low = st.predicate.replace('_', " ");
                    let t_low = st.target_concept.to_lowercase().replace('_', " ");
                    format!("{} {} {}", s_low, p_low, t_low)
                })
                .collect();
            let mut text = format!("Causal wave from {}: {}.", root_low, clauses.join(" -> "));
            if let Some(first) = text.get_mut(0..1) {
                first.make_ascii_uppercase();
            }
            text
        };

        CausalSimulationDto {
            root_concept: clean_root,
            affected_concepts: affected_set,
            causal_steps,
            causal_synthesis,
        }
    }

    /// Phase 18: Active Curiosity-Driven Autonomous Hypothesis Promotion (`autonomous_curiosity_step`).
    ///
    /// Discovers epistemic hypotheses via `discover_epistemic_hypotheses`, filters for those
    /// meeting `min_confidence` and passing `verify_spoke_invariants` (`I_DAG` + `I_MUTEX`),
    /// and promotes up to `max_promotions` into the 360° Spider-Web Arena as verified analogical
    /// synapses while generating natural-language inquiry questions for unresolved curiosity gaps.
    pub fn autonomous_curiosity_step(
        &mut self,
        min_confidence: f32,
        max_promotions: usize,
    ) -> AutonomousCuriosityStepDto {
        let report = self.discover_epistemic_hypotheses(max_promotions.max(8) * 2);
        let mut promoted: Vec<AnalogicalHypothesisDto> = Vec::new();
        let cap = max_promotions.max(1);

        for hyp in &report.analogical_hypotheses {
            if promoted.len() >= cap {
                break;
            }
            if hyp.confidence < min_confidence {
                continue;
            }
            let (passed, _gate) = self.verify_spoke_invariants(
                &hyp.subject_concept,
                &hyp.predicate,
                &hyp.target_concept,
            );
            if passed {
                self.learn_spoke_relation(
                    &hyp.subject_concept,
                    &hyp.predicate,
                    &hyp.target_concept,
                    0.18 * hyp.confidence,
                    true,
                );
                promoted.push(hyp.clone());
            }
        }

        let open_questions: Vec<String> = report
            .curiosity_gaps
            .iter()
            .take(5)
            .map(|gap| {
                let c_low = gap.concept_id.to_lowercase().replace('_', " ");
                let first_missing = gap
                    .missing_spokes
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "properties".to_string());
                format!("What are the {} characteristics of {}?", first_missing, c_low)
            })
            .collect();

        AutonomousCuriosityStepDto {
            hypotheses_evaluated: report.analogical_hypotheses.len(),
            hypotheses_promoted: promoted.len(),
            promoted_hypotheses: promoted,
            open_questions,
        }
    }

    /// Phase 19: Fast Native Rust Snapshot Persistence (`save_arena_snapshot` / `load_arena_snapshot`).
    pub fn save_arena_snapshot(&self, path_str: &str) -> SnapshotStatusDto {
        let path = std::path::Path::new(path_str);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_vec(self) {
            Ok(bytes) => {
                let n_bytes = bytes.len();
                match std::fs::write(path, bytes) {
                    Ok(_) => SnapshotStatusDto {
                        success: true,
                        path: path_str.to_string(),
                        total_cells: self.cells.len(),
                        bytes_written: n_bytes,
                    },
                    Err(_) => SnapshotStatusDto {
                        success: false,
                        path: path_str.to_string(),
                        total_cells: self.cells.len(),
                        bytes_written: 0,
                    },
                }
            }
            Err(_) => SnapshotStatusDto {
                success: false,
                path: path_str.to_string(),
                total_cells: self.cells.len(),
                bytes_written: 0,
            },
        }
    }

    pub fn load_arena_snapshot(&mut self, path_str: &str) -> SnapshotStatusDto {
        let path = std::path::Path::new(path_str);
        let Ok(bytes) = std::fs::read(path) else {
            return SnapshotStatusDto {
                success: false,
                path: path_str.to_string(),
                total_cells: self.cells.len(),
                bytes_written: 0,
            };
        };
        let n_bytes = bytes.len();
        let Ok(restored) = serde_json::from_slice::<SpiderWebArena>(&bytes) else {
            return SnapshotStatusDto {
                success: false,
                path: path_str.to_string(),
                total_cells: self.cells.len(),
                bytes_written: 0,
            };
        };
        *self = restored;
        SnapshotStatusDto {
            success: true,
            path: path_str.to_string(),
            total_cells: self.cells.len(),
            bytes_written: n_bytes,
        }
    }

    /// Phase 20: Native Rust Counterfactual Intervention Wave (`intervene_counterfactual_360`).
    ///
    /// Evaluates the causal/procedural wave from `root_concept` with `suppressed_concept`
    /// virtually clamped out, comparing the baseline wave against the intervened wave to
    /// identify `invalidated_conclusions` (downstream states that collapse when `suppressed_concept`
    /// is removed) and `preserved_conclusions` (downstream states reachable via alternate paths).
    pub fn intervene_counterfactual_360(
        &mut self,
        root_concept: &str,
        suppressed_concept: &str,
        max_hops: usize,
    ) -> CounterfactualInterventionDto {
        let clean_root = root_concept.trim().to_uppercase().replace(' ', "_");
        let clean_supp = suppressed_concept.trim().to_uppercase().replace(' ', "_");
        let hops = max_hops.clamp(1, 5);

        let baseline = self.simulate_causal_wave_360(&clean_root, hops);
        if baseline.affected_concepts.is_empty() || clean_root == clean_supp {
            let invalidated = if clean_root == clean_supp {
                baseline.affected_concepts.clone()
            } else {
                Vec::new()
            };
            return CounterfactualInterventionDto {
                root_concept: clean_root.clone(),
                suppressed_concept: clean_supp.clone(),
                baseline_affected: baseline.affected_concepts,
                invalidated_conclusions: invalidated,
                preserved_conclusions: Vec::new(),
                counterfactual_synthesis: format!(
                    "If {} is suppressed, the wave from {} is blocked at the source.",
                    clean_supp.to_lowercase().replace('_', " "),
                    clean_root.to_lowercase().replace('_', " ")
                ),
            };
        }

        // Propagate virtual intervened wave skipping suppressed_concept
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(clean_root.clone());
        visited.insert(clean_supp.clone());
        let mut preserved_set: Vec<String> = Vec::new();

        if let Some(&start_id) = self.name_to_id.get(&clean_root) {
            let mut frontier: Vec<u32> = vec![start_id];
            for _hop in 1..=hops {
                let mut next_frontier: Vec<u32> = Vec::new();
                for cid in &frontier {
                    let Some(cell) = self.cells.get(*cid as usize) else {
                        continue;
                    };
                    for edges in &cell.spoke_synapses {
                        for syn in edges {
                            if syn.pos_evidence <= syn.neg_evidence
                                || syn.predicate == "co_occurs_with"
                                || syn.predicate == "disjoint_with"
                                || syn.target_name == clean_supp
                            {
                                continue;
                            }
                            if !preserved_set.contains(&syn.target_name)
                                && syn.target_name != clean_root
                            {
                                preserved_set.push(syn.target_name.clone());
                            }
                            if visited.insert(syn.target_name.clone()) {
                                if let Some(&tid) = self.name_to_id.get(&syn.target_name) {
                                    next_frontier.push(tid);
                                }
                            }
                        }
                    }
                }
                if next_frontier.is_empty() {
                    break;
                }
                frontier = next_frontier;
            }
        }

        let mut invalidated: Vec<String> = Vec::new();
        for c in &baseline.affected_concepts {
            if !preserved_set.contains(c) {
                invalidated.push(c.clone());
            }
        }

        let root_low = clean_root.to_lowercase().replace('_', " ");
        let supp_low = clean_supp.to_lowercase().replace('_', " ");
        let synthesis = if invalidated.is_empty() {
            format!(
                "Counterfactual intervention: suppressing {} leaves all downstream effects of {} intact.",
                supp_low, root_low
            )
        } else {
            let inv_list: Vec<String> = invalidated
                .iter()
                .map(|s| s.to_lowercase().replace('_', " "))
                .collect();
            format!(
                "Counterfactual intervention: if {} is prevented following {}, downstream conclusions [{}] are invalidated.",
                supp_low,
                root_low,
                inv_list.join(", ")
            )
        };

        CounterfactualInterventionDto {
            root_concept: clean_root,
            suppressed_concept: clean_supp,
            baseline_affected: baseline.affected_concepts,
            invalidated_conclusions: invalidated,
            preserved_conclusions: preserved_set,
            counterfactual_synthesis: synthesis,
        }
    }

    /// Phase 21: Native Rust Shortest 360° Relational Path Bridge (`find_relational_bridge_360`).
    ///
    /// Performs a breadth-first relational wave search across all 8 harmonic spokes (traversing
    /// both forward and inverse verified synapses) to discover the shortest high-confidence
    /// multi-hop connecting chain between `concept_a` and `concept_b`.
    pub fn find_relational_bridge_360(
        &self,
        concept_a: &str,
        concept_b: &str,
        max_hops: usize,
    ) -> RelationalBridgeDto {
        let clean_a = concept_a.trim().to_uppercase().replace(' ', "_");
        let clean_b = concept_b.trim().to_uppercase().replace(' ', "_");
        let hops = max_hops.clamp(1, 6);

        if !self.name_to_id.contains_key(&clean_a) || !self.name_to_id.contains_key(&clean_b) {
            return RelationalBridgeDto {
                concept_a: clean_a,
                concept_b: clean_b,
                connected: false,
                hop_count: 0,
                bridge_steps: Vec::new(),
                bridge_synthesis: String::new(),
            };
        }
        if clean_a == clean_b {
            return RelationalBridgeDto {
                concept_a: clean_a.clone(),
                concept_b: clean_b,
                connected: true,
                hop_count: 0,
                bridge_steps: Vec::new(),
                bridge_synthesis: format!(
                    "{} is identical to itself.",
                    clean_a.to_lowercase().replace('_', " ")
                ),
            };
        }

        // Build bidirectional adjacency of verified non-co-occurrence edges: node -> Vec<(neighbor, predicate, spoke, direction, conf)>
        let mut adj: std::collections::HashMap<
            String,
            Vec<(String, String, String, String, f32)>,
        > = std::collections::HashMap::new();
        for cell in &self.cells {
            if cell.concept_id == "CORE_ROOT" || cell.concept_id.starts_with("SUBCORE_") {
                continue;
            }
            for (s_idx, edges) in cell.spoke_synapses.iter().enumerate() {
                for syn in edges {
                    if syn.pos_evidence <= syn.neg_evidence
                        || syn.predicate == "co_occurs_with"
                        || syn.predicate == "disjoint_with"
                    {
                        continue;
                    }
                    let total_ev = syn.pos_evidence + syn.neg_evidence;
                    let conf = ((total_ev / (total_ev + 1.0)) * 10000.0).round() / 10000.0;
                    let spoke_name = SPOKE_NAMES[s_idx].to_string();
                    adj.entry(cell.concept_id.clone()).or_default().push((
                        syn.target_name.clone(),
                        syn.predicate.clone(),
                        spoke_name.clone(),
                        "->".to_string(),
                        conf,
                    ));
                    adj.entry(syn.target_name.clone()).or_default().push((
                        cell.concept_id.clone(),
                        syn.predicate.clone(),
                        spoke_name,
                        "<-".to_string(),
                        conf,
                    ));
                }
            }
        }

        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(clean_a.clone());
        let mut queue: std::collections::VecDeque<(String, Vec<BridgeStepDto>)> =
            std::collections::VecDeque::new();
        queue.push_back((clean_a.clone(), Vec::new()));

        while let Some((curr, path)) = queue.pop_front() {
            if path.len() >= hops {
                continue;
            }
            let Some(neighbors) = adj.get(&curr) else {
                continue;
            };
            for (next_node, pred, spoke, dir, conf) in neighbors {
                let mut next_path = path.clone();
                next_path.push(BridgeStepDto {
                    from_concept: curr.clone(),
                    predicate: pred.clone(),
                    to_concept: next_node.clone(),
                    spoke: spoke.clone(),
                    direction: dir.clone(),
                    confidence: *conf,
                });
                if next_node == &clean_b {
                    let clauses: Vec<String> = next_path
                        .iter()
                        .map(|st| {
                            let f_low = st.from_concept.to_lowercase().replace('_', " ");
                            let p_low = st.predicate.replace('_', " ");
                            let t_low = st.to_concept.to_lowercase().replace('_', " ");
                            if st.direction == "->" {
                                format!("{} {} {}", f_low, p_low, t_low)
                            } else {
                                format!("{} {} {}", t_low, p_low, f_low)
                            }
                        })
                        .collect();
                    let synthesis = format!(
                        "Relational bridge ({} hops): {}.",
                        next_path.len(),
                        clauses.join(" ; ")
                    );
                    return RelationalBridgeDto {
                        concept_a: clean_a,
                        concept_b: clean_b,
                        connected: true,
                        hop_count: next_path.len(),
                        bridge_steps: next_path,
                        bridge_synthesis: synthesis,
                    };
                }
                if visited.insert(next_node.clone()) {
                    queue.push_back((next_node.clone(), next_path));
                }
            }
        }

        RelationalBridgeDto {
            concept_a: clean_a,
            concept_b: clean_b,
            connected: false,
            hop_count: 0,
            bridge_steps: Vec::new(),
            bridge_synthesis: String::new(),
        }
    }

    /// Phase 22: Native Rust 360° Spider-Web Topology & Mermaid Exporter (`export_topology_graph`).
    pub fn export_topology_graph(&self, max_edges: usize) -> TopologyGraphDto {
        let limit = max_edges.max(1);
        let mut nodes: Vec<String> = Vec::new();
        let mut edges: Vec<BridgeStepDto> = Vec::new();
        let mut mermaid_lines: Vec<String> = vec!["flowchart TD".to_string()];

        for cell in &self.cells {
            nodes.push(cell.concept_id.clone());
            for (s_idx, spoke_edges) in cell.spoke_synapses.iter().enumerate() {
                for syn in spoke_edges {
                    if syn.pos_evidence <= syn.neg_evidence || syn.predicate == "co_occurs_with" {
                        continue;
                    }
                    if edges.len() >= limit {
                        break;
                    }
                    let total_ev = syn.pos_evidence + syn.neg_evidence;
                    let conf = ((total_ev / (total_ev + 1.0)) * 10000.0).round() / 10000.0;
                    edges.push(BridgeStepDto {
                        from_concept: cell.concept_id.clone(),
                        predicate: syn.predicate.clone(),
                        to_concept: syn.target_name.clone(),
                        spoke: SPOKE_NAMES[s_idx].to_string(),
                        direction: "->".to_string(),
                        confidence: conf,
                    });
                    mermaid_lines.push(format!(
                        "    {}[\"{}\"] -->|\"{}\"| {}[\"{}\"]",
                        cell.concept_id,
                        cell.concept_id,
                        syn.predicate,
                        syn.target_name,
                        syn.target_name
                    ));
                }
            }
        }

        TopologyGraphDto {
            total_cells: self.cells.len(),
            total_verified_edges: edges.len(),
            nodes,
            edges,
            mermaid_diagram: mermaid_lines.join("\n"),
        }
    }

    /// Phase 23: Native Rust Emergent Concept Abstraction & Centroid Clustering (`discover_latent_clusters_360`).
    ///
    /// Groups active concept cells that share high-confidence positive attractors (`target_name`)
    /// into emergent latent clusters, computing 64-D centroid coherence (`mean pairwise cosine similarity`)
    /// and identifying candidate domain subcore abstractions.
    pub fn discover_latent_clusters_360(&self, min_members: usize) -> Vec<LatentClusterDto> {
        let req_members = min_members.max(2);
        let mut hub_to_members: std::collections::HashMap<
            (String, String, String),
            Vec<(usize, String)>,
        > = std::collections::HashMap::new();

        for (idx, cell) in self.cells.iter().enumerate() {
            if cell.concept_id == "CORE_ROOT" || cell.concept_id.starts_with("SUBCORE_") {
                continue;
            }
            for (s_idx, spoke_edges) in cell.spoke_synapses.iter().enumerate() {
                for syn in spoke_edges {
                    if syn.pos_evidence <= syn.neg_evidence
                        || syn.predicate == "co_occurs_with"
                        || syn.predicate == "disjoint_with"
                    {
                        continue;
                    }
                    let key = (
                        syn.target_name.clone(),
                        syn.predicate.clone(),
                        SPOKE_NAMES[s_idx].to_string(),
                    );
                    let entry = hub_to_members.entry(key).or_default();
                    if !entry.iter().any(|(_, cid)| cid == &cell.concept_id) {
                        entry.push((idx, cell.concept_id.clone()));
                    }
                }
            }
        }

        let mut clusters: Vec<LatentClusterDto> = Vec::new();
        for ((attractor, predicate, spoke), members) in hub_to_members {
            if members.len() < req_members {
                continue;
            }
            // Compute centroid in 64-D space and mean cosine coherence
            let mut centroid = [0.0f32; MLU_DIM];
            for (c_idx, _) in &members {
                let anchor = self.get_anchor_array(*c_idx as u32);
                for d in 0..MLU_DIM {
                    centroid[d] += anchor[d];
                }
            }
            normalize_in_place(&mut centroid);

            let mut total_coh = 0.0f32;
            for (c_idx, _) in &members {
                let anchor = self.get_anchor_array(*c_idx as u32);
                total_coh += dot_product(&anchor, &centroid).clamp(0.0, 1.0);
            }
            let coherence =
                ((total_coh / (members.len() as f32)) * 10000.0).round() / 10000.0;
            let member_names: Vec<String> = members.into_iter().map(|(_, n)| n).collect();

            clusters.push(LatentClusterDto {
                shared_attractor: attractor.clone(),
                shared_predicate: predicate,
                spoke,
                member_count: member_names.len(),
                member_concepts: member_names,
                centroid_coherence: coherence,
                emergent_abstraction_id: format!("CLUSTER_{}", attractor),
            });
        }

        clusters.sort_by(|a, b| {
            b.member_count
                .cmp(&a.member_count)
                .then_with(|| {
                    b.centroid_coherence
                        .partial_cmp(&a.centroid_coherence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        clusters
    }

    /// Phase 24: Native Rust Non-Monotonic Belief Revision & Evidence Update (`revise_belief_360`).
    ///
    /// Applies explicit positive (`new_verified = true`) or refuting (`new_verified = false`)
    /// evidence of strength `evidence_weight` to `(subject, predicate, object)`, flipping
    /// synapse polarity when refuting evidence outweighs prior positive evidence and
    /// returning full NARS truth-value diagnostics (`BeliefRevisionDto`).
    pub fn revise_belief_360(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        new_verified: bool,
        evidence_weight: f32,
    ) -> BeliefRevisionDto {
        let sub_clean = subject.trim().to_uppercase().replace(' ', "_");
        let obj_clean = object.trim().to_uppercase().replace(' ', "_");
        let pred_clean = predicate.trim().to_lowercase();
        let weight = evidence_weight.clamp(0.5, 20.0);

        let pred_vec = deterministic_seed_vec(&pred_clean);
        let obj_seed = deterministic_seed_vec(&obj_clean);
        let spoke_idx = route_phase_spoke_idx(&pred_vec, &obj_seed);
        let default_hub = DEFAULT_SUBCORES[spoke_idx].to_string();
        let sub_id = self.ensure_cell(&sub_clean, "leaf", Some(default_hub.clone()));
        let obj_id = self.ensure_cell(&obj_clean, "leaf", Some(default_hub));

        let cell = &mut self.cells[sub_id as usize];
        let mut old_pos = 0.0f32;
        let mut old_neg = 0.0f32;

        // Locate existing synapse across all spokes on subject cell
        let mut found_spoke = spoke_idx;
        let mut found_pos_idx: Option<usize> = None;
        for (s_i, edges) in cell.spoke_synapses.iter().enumerate() {
            if let Some(p_i) = edges
                .iter()
                .position(|s| s.target_name == obj_clean && s.predicate == pred_clean)
            {
                found_spoke = s_i;
                found_pos_idx = Some(p_i);
                old_pos = edges[p_i].pos_evidence;
                old_neg = edges[p_i].neg_evidence;
                break;
            }
        }

        let old_polarity = if old_pos == 0.0 && old_neg == 0.0 {
            0
        } else if old_pos >= old_neg {
            1
        } else {
            -1
        };

        let (new_pos, new_neg) = if let Some(p_i) = found_pos_idx {
            let syn = &mut cell.spoke_synapses[found_spoke][p_i];
            if new_verified {
                syn.pos_evidence += weight;
                syn.affinity = (syn.affinity + 0.35 * weight).min(3.5);
            } else {
                syn.neg_evidence += weight;
                syn.affinity = (syn.affinity - 0.45 * weight).max(-3.5);
            }
            (syn.pos_evidence, syn.neg_evidence)
        } else {
            let pos = if new_verified { weight } else { 0.0 };
            let neg = if new_verified { 0.0 } else { weight };
            let aff = if new_verified {
                (0.35 * weight).min(3.5)
            } else {
                (-0.45 * weight).max(-3.5)
            };
            cell.spoke_synapses[spoke_idx].push(SpokeSynapse {
                target_id: obj_id,
                target_name: obj_clean.clone(),
                predicate: pred_clean.clone(),
                affinity: aff,
                pos_evidence: pos,
                neg_evidence: neg,
            });
            (pos, neg)
        };

        let new_polarity = if new_pos >= new_neg { 1 } else { -1 };
        let polarity_flipped = old_polarity != 0 && old_polarity != new_polarity;
        let total_ev = new_pos + new_neg;
        let nars_conf = ((total_ev / (total_ev + 1.0)) * 10000.0).round() / 10000.0;

        BeliefRevisionDto {
            subject_concept: sub_clean,
            predicate: pred_clean,
            target_concept: obj_clean,
            spoke: SPOKE_NAMES[found_spoke].to_string(),
            old_polarity,
            new_polarity,
            polarity_flipped,
            pos_evidence: new_pos,
            neg_evidence: new_neg,
            nars_confidence: nars_conf,
        }
    }

    /// Phase 30: Native Rust System-2 Multi-Wave Deliberative Thinking (`deliberate_360`).
    ///
    /// Executes a 5-stage deliberative wave loop over `concept_id`:
    ///   1. 360° Taxonomic, Capability & Meronymic Beam Wave (`query_360_beam`)
    ///   2. Multi-Hop Causal & Forward Consequence Wave (`simulate_causal_wave_360`)
    ///   3. Emergent Cluster & Analogical Peer Resonance Wave
    ///   4. Epistemic Free-Energy & Spoke Gap Assessment
    ///   5. Coherent Multi-Sentence Discourse Synthesis
    pub fn deliberate_360(&mut self, concept_id: &str, max_hops: usize) -> Deliberation360Dto {
        let hops = max_hops.clamp(1, 5);
        let beam = self.query_360_beam(concept_id, hops, 12);
        let clean = beam.focal_concept.clone();

        if beam.role == "unknown" {
            let subj_low = clean.to_lowercase().replace('_', " ");
            return Deliberation360Dto {
                focal_concept: clean,
                waves_executed: 1,
                free_energy: 1.0,
                verified_facts: Vec::new(),
                causal_consequences: Vec::new(),
                cluster_peers: Vec::new(),
                analogical_insights: Vec::new(),
                follow_up_question: format!("What is {} and how does it relate to known concepts?", subj_low),
                deliberative_synthesis: format!(
                    "I do not yet have verified 360° MLU knowledge about {}. What is {} and how does it relate to known concepts?",
                    subj_low, subj_low
                ),
            };
        }

        // Wave 1: Verified facts
        let verified_facts: Vec<SpokePredictionDto> = beam
            .predictions
            .iter()
            .filter(|p| p.verified && p.predicate != "co_occurs_with")
            .cloned()
            .collect();

        // Wave 2: Causal wave from focal concept and its direct procedural targets
        let causal_sim = self.simulate_causal_wave_360(&clean, hops);
        let causal_consequences = causal_sim.affected_concepts.clone();

        // Wave 3: Cluster peers & Analogical insights
        let clusters = self.discover_latent_clusters_360(2);
        let mut cluster_peers: Vec<String> = Vec::new();
        for cl in &clusters {
            if cl.member_concepts.contains(&clean) {
                for m in &cl.member_concepts {
                    if m != &clean && !cluster_peers.contains(m) {
                        cluster_peers.push(m.clone());
                    }
                }
            }
        }

        let curiosity = self.discover_epistemic_hypotheses(15);
        let analogical_insights: Vec<AnalogicalHypothesisDto> = curiosity
            .analogical_hypotheses
            .into_iter()
            .filter(|h| h.subject_concept == clean)
            .take(3)
            .collect();

        // Wave 4: Epistemic Free-Energy & Curiosity Question
        let avg_entropy = if verified_facts.is_empty() {
            1.0
        } else {
            let sum_h: f32 = verified_facts.iter().map(|p| p.epistemic_entropy).sum();
            sum_h / (verified_facts.len() as f32)
        };
        let info_bonus = 0.15 * ((1.0 + verified_facts.len() as f32).ln())
            + 0.10 * ((1.0 + causal_consequences.len() as f32).ln())
            + 0.08 * ((1.0 + cluster_peers.len() as f32).ln());
        let free_energy = (((avg_entropy - info_bonus).clamp(0.0, 1.0) * 10000.0).round()) / 10000.0;

        let subj_low = clean.to_lowercase().replace('_', " ");
        let follow_up_question = curiosity
            .curiosity_gaps
            .iter()
            .find(|g| g.concept_id == clean)
            .and_then(|g| g.missing_spokes.first())
            .map(|spoke| format!("What are the {} characteristics of {}?", spoke, subj_low))
            .unwrap_or_default();

        // Wave 5: Coherent Multi-Sentence Discourse Synthesis
        let mut sentences: Vec<String> = Vec::new();
        if !beam.fluent_synthesis.is_empty() {
            sentences.push(beam.fluent_synthesis.clone());
        }
        if !causal_consequences.is_empty() && !causal_sim.causal_synthesis.is_empty() {
            sentences.push(causal_sim.causal_synthesis);
        }
        if !cluster_peers.is_empty() {
            let peer_list: Vec<String> = cluster_peers
                .iter()
                .take(3)
                .map(|p| p.to_lowercase().replace('_', " "))
                .collect();
            if let Some(first_hyp) = analogical_insights.first() {
                sentences.push(format!(
                    "Through 360° cluster resonance with {}, {} may also {} {} (shared {}).",
                    peer_list.join(", "),
                    subj_low,
                    first_hyp.predicate.replace('_', " "),
                    first_hyp.target_concept.to_lowercase().replace('_', " "),
                    first_hyp.shared_attractor.to_lowercase().replace('_', " ")
                ));
            } else {
                sentences.push(format!(
                    "In the 360° arena, {} clusters with {}.",
                    subj_low,
                    peer_list.join(", ")
                ));
            }
        }
        if !follow_up_question.is_empty() && verified_facts.len() < 4 {
            sentences.push(follow_up_question.clone());
        }

        Deliberation360Dto {
            focal_concept: clean,
            waves_executed: 5,
            free_energy,
            verified_facts,
            causal_consequences,
            cluster_peers,
            analogical_insights,
            follow_up_question,
            deliberative_synthesis: sentences.join(" "),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamTrainStatsDto {
    pub sentences_processed: usize,
    pub tokens_processed: usize,
    pub spoke_updates: usize,
    pub average_surprise: f32,
    pub total_cells: usize,
    pub mitosis_promotions: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Query360BeamDto {
    pub focal_concept: String,
    pub active_subcore: String,
    pub role: String,
    pub activated_cells: Vec<String>,
    pub predictions: Vec<SpokePredictionDto>,
    pub generated_continuation: String,
    pub fluent_synthesis: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsolidationReportDto {
    pub replayed_edges: usize,
    pub pruned_transitions: usize,
    pub average_replay_surprise: f32,
    pub mitosis_promotions: usize,
    pub total_cells: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedAttractorDto {
    pub predicate: String,
    pub target_concept: String,
    pub spoke: String,
    pub combined_confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Compare360Dto {
    pub concept_a: String,
    pub concept_b: String,
    pub anchor_similarity: f32,
    pub mutually_exclusive: bool,
    pub shared_attractors: Vec<SharedAttractorDto>,
    pub distinct_a: Vec<SpokePredictionDto>,
    pub distinct_b: Vec<SpokePredictionDto>,
    pub comparative_synthesis: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalogicalHypothesisDto {
    pub subject_concept: String,
    pub predicate: String,
    pub target_concept: String,
    pub spoke: String,
    pub analog_source: String,
    pub shared_attractor: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CuriosityGapDto {
    pub concept_id: String,
    pub populated_spokes: usize,
    pub spoke_sparsity: f32,
    pub missing_spokes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CuriosityReportDto {
    pub analogical_hypotheses: Vec<AnalogicalHypothesisDto>,
    pub curiosity_gaps: Vec<CuriosityGapDto>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalStepDto {
    pub hop: usize,
    pub source_concept: String,
    pub predicate: String,
    pub target_concept: String,
    pub spoke: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalSimulationDto {
    pub root_concept: String,
    pub affected_concepts: Vec<String>,
    pub causal_steps: Vec<CausalStepDto>,
    pub causal_synthesis: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutonomousCuriosityStepDto {
    pub hypotheses_evaluated: usize,
    pub hypotheses_promoted: usize,
    pub promoted_hypotheses: Vec<AnalogicalHypothesisDto>,
    pub open_questions: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotStatusDto {
    pub success: bool,
    pub path: String,
    pub total_cells: usize,
    pub bytes_written: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CounterfactualInterventionDto {
    pub root_concept: String,
    pub suppressed_concept: String,
    pub baseline_affected: Vec<String>,
    pub invalidated_conclusions: Vec<String>,
    pub preserved_conclusions: Vec<String>,
    pub counterfactual_synthesis: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeStepDto {
    pub from_concept: String,
    pub predicate: String,
    pub to_concept: String,
    pub spoke: String,
    pub direction: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelationalBridgeDto {
    pub concept_a: String,
    pub concept_b: String,
    pub connected: bool,
    pub hop_count: usize,
    pub bridge_steps: Vec<BridgeStepDto>,
    pub bridge_synthesis: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyGraphDto {
    pub total_cells: usize,
    pub total_verified_edges: usize,
    pub nodes: Vec<String>,
    pub edges: Vec<BridgeStepDto>,
    pub mermaid_diagram: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentClusterDto {
    pub shared_attractor: String,
    pub shared_predicate: String,
    pub spoke: String,
    pub member_count: usize,
    pub member_concepts: Vec<String>,
    pub centroid_coherence: f32,
    pub emergent_abstraction_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BeliefRevisionDto {
    pub subject_concept: String,
    pub predicate: String,
    pub target_concept: String,
    pub spoke: String,
    pub old_polarity: i32,
    pub new_polarity: i32,
    pub polarity_flipped: bool,
    pub pos_evidence: f32,
    pub neg_evidence: f32,
    pub nars_confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Deliberation360Dto {
    pub focal_concept: String,
    pub waves_executed: usize,
    pub free_energy: f32,
    pub verified_facts: Vec<SpokePredictionDto>,
    pub causal_consequences: Vec<String>,
    pub cluster_peers: Vec<String>,
    pub analogical_insights: Vec<AnalogicalHypothesisDto>,
    pub follow_up_question: String,
    pub deliberative_synthesis: String,
}


