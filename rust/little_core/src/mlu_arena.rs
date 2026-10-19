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

pub fn map_predicate_to_spoke_idx(predicate: &str) -> usize {
    let norm = predicate.trim().to_lowercase().replace(' ', "_");
    match norm.as_str() {
        "is_a" | "instance_of" | "subclass_of" | "kind_of" | "disjoint_with" => 0,
        "part_of" | "has_part" | "has" | "contains" | "made_of" => 1,
        "has_property" | "is" | "color" | "shape" | "taste" | "temperature" => 2,
        "causes" | "becomes" | "requires" | "produces" | "triggers" => 3,
        "can" | "capable_of" | "used_for" | "performs" => 4,
        "located_in" | "found_in" | "lives_in" | "co_occurs_with" => 5,
        _ => {
            if norm.contains("part") || norm.contains("contain") || norm.contains("consist") {
                1
            } else if norm.contains("prop") || norm.contains("color") || norm.contains("size") {
                2
            } else if norm.contains("cause") || norm.contains("change") || norm.contains("req") {
                3
            } else if norm.contains("can") || norm.contains("use") || norm.contains("skill") {
                4
            } else if norm.contains("locat") || norm.contains("live") || norm.contains("find") {
                5
            } else {
                0
            }
        }
    }
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
        for s_name in SPOKE_NAMES {
            spoke_weights.push(deterministic_seed_vec(&format!("SPOKE::{s_name}")).to_vec());
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
        let spoke_idx = map_predicate_to_spoke_idx(predicate);
        let default_hub = DEFAULT_SUBCORES[spoke_idx].to_string();
        let sub_id = self.ensure_cell(subject, "leaf", Some(default_hub.clone()));
        let obj_id = self.ensure_cell(object, "leaf", Some(default_hub));

        let sub_anchor = self.get_anchor_array(sub_id);
        let obj_anchor = self.get_anchor_array(obj_id);

        // Step Wave 1 (LFM Liquid State) on subject, object, and CORE_ROOT
        if let Some(sub_cell) = self.cells.get_mut(sub_id as usize) {
            sub_cell.forward_liquid_step(&obj_anchor, &sub_anchor, &obj_anchor, 1.0);
        }
        if let Some(obj_cell) = self.cells.get_mut(obj_id as usize) {
            obj_cell.forward_liquid_step(&sub_anchor, &obj_anchor, &sub_anchor, 1.0);
        }
        if let Some(core_cell) = self.cells.get_mut(0) {
            core_cell.forward_liquid_step(&sub_anchor, &obj_anchor, &sub_anchor, 1.0);
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

    /// Wave 2: Predict top-k 360° spoke targets for `concept_id` on `spoke_idx` in Rust.
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
        if spoke_idx >= NUM_SPOKES || cell.spoke_synapses[spoke_idx].is_empty() {
            return Vec::new();
        }

        let mut raw_scores: Vec<(String, f32, f32, String, bool)> = Vec::new();
        for syn in &cell.spoke_synapses[spoke_idx] {
            if syn.neg_evidence > syn.pos_evidence {
                continue;
            }
            let target_vec = self.get_anchor_array(syn.target_id);
            let neural_score = cell.score_spoke_target(spoke_idx, &syn.target_name, &target_vec);
            let total_ev = syn.pos_evidence + syn.neg_evidence;
            let nars_conf = (syn.pos_evidence / total_ev.max(1.0)) * (total_ev / (total_ev + 0.5));
            raw_scores.push((
                syn.target_name.clone(),
                neural_score,
                nars_conf,
                syn.predicate.clone(),
                syn.pos_evidence > 0.0,
            ));
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

        let spoke_label = SPOKE_NAMES[spoke_idx].to_string();
        let mut results: Vec<SpokePredictionDto> = raw_scores
            .into_iter()
            .map(|(target_name, neural_score, nars_conf, pred_label, is_ver)| {
                let combined = 0.65 * nars_conf + 0.35 * neural_score;
                SpokePredictionDto {
                    source_concept: cell.concept_id.clone(),
                    spoke: spoke_label.clone(),
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

    /// Pure-Rust 360° Liquid-Resonance Stream Learner (zero hardcoded word lists or regexes).
    ///
    /// Slides a 3-token LFM-2.5 short-convolution window across raw UTF-8 text, selects the
    /// radial spoke (0..6) with highest liquid trajectory resonance, updates local JEPA spoke
    /// weights + Multi-Token Prediction (MTP) transitions, and triggers 5-Brake Fractal Mitosis.
    pub fn train_raw_stream(&mut self, text: &str, learning_rate: f32) -> StreamTrainStatsDto {
        let mut sentences_processed: usize = 0;
        let mut tokens_processed: usize = 0;
        let mut spoke_updates: usize = 0;
        let mut total_surprise: f32 = 0.0;
        let mut surprise_count: usize = 0;
        let initial_mitosis = self.total_mitosis_events;

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

            // 2. Slide 3-tap LFM-2.5 liquid window over content tokens (len >= 3)
            let content_tokens: Vec<String> = tokens
                .iter()
                .filter(|w| w.len() >= 3)
                .cloned()
                .collect();

            for idx in 0..content_tokens.len() {
                let curr_tok = &content_tokens[idx];
                let prev1 = if idx >= 1 { &content_tokens[idx - 1] } else { curr_tok };
                let prev2 = if idx >= 2 { &content_tokens[idx - 2] } else { prev1 };

                let cid = self.ensure_cell(curr_tok, "leaf", None);
                let v1 = deterministic_seed_vec(curr_tok);
                let v2 = deterministic_seed_vec(prev1);
                let v3 = deterministic_seed_vec(prev2);

                if let Some(cell) = self.cells.get_mut(cid as usize) {
                    cell.forward_liquid_step(&v1, &v2, &v3, 1.0);
                    cell.learn_token_sequence(&tokens, learning_rate * 0.8);
                }

                if idx + 1 < content_tokens.len() {
                    let next_tok = &content_tokens[idx + 1];
                    if next_tok != curr_tok {
                        let next_vec = deterministic_seed_vec(next_tok);
                        // Automatically route to the 360° radial spoke with highest liquid resonance
                        let mut best_spoke: usize = 0;
                        let mut best_res: f32 = -1e9;
                        if let Some(cell) = self.cells.get(cid as usize) {
                            for s_idx in 0..NUM_SPOKES {
                                let proj = cell.project_spoke(s_idx);
                                let res = dot_product(&proj, &next_vec);
                                if res > best_res {
                                    best_res = res;
                                    best_spoke = s_idx;
                                }
                            }
                        }
                        let spoke_pred = SPOKE_NAMES[best_spoke];
                        let s = self.learn_spoke_relation(
                            curr_tok,
                            spoke_pred,
                            next_tok,
                            learning_rate,
                            true,
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


