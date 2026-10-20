//! True End-to-End Differentiable 360° Spider-Web Neural MLU Network (`NeuralSpiderWebMlu`).
//!
//! 100% Neural Architecture with Analytical Backpropagation & AdamW Optimizer in Rust:
//! 1. Trainable Token Embeddings `E (V x D)` + Positional Embeddings `P (T_max x D)`
//! 2. Causal Liquid Time-Constant Self-Attention (`W_Q, W_K, W_V, W_tau, b_tau`)
//! 3. 360° Softmax Spoke Router (`W_route, b_route`) over `K = 6` Differentiable Neural Logic MLU Cells
//! 4. Differentiable Fuzzy/Product T-Norm Logic Gate Neurons inside each MLU Cell:
//!    - `AND(a, b) = a * b`
//!    - `OR(a, b) = a + b - a * b`
//!    - `IMPLIES(a, b) = 1 - a + a * b`
//!    - `XOR(a, b) = a + b - 2 * a * b`
//! 5. Autoregressive Neural Vocabulary Head (`W_lm, b_lm`) generating English token-by-token
//!    purely from neural hidden states (zero string templates, zero regexes, zero phrase dictionaries).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const D_MODEL: usize = 48;
pub const MAX_SEQ_LEN: usize = 64;
pub const NUM_LOGIC_CELLS: usize = 6;
pub const NUM_LOGIC_GATES: usize = 4; // 0: AND, 1: OR, 2: IMPLIES, 3: XOR

pub const SPOKE_CELL_NAMES: [&str; NUM_LOGIC_CELLS] = [
    "CELL_0_TAXONOMY",
    "CELL_1_MEREOLOGY",
    "CELL_2_CAUSALITY",
    "CELL_3_SPATIAL",
    "CELL_4_CAPABILITY",
    "CELL_5_DYNAMICS",
];

pub const LOGIC_GATE_NAMES: [&str; NUM_LOGIC_GATES] = ["AND", "OR", "IMPLIES", "XOR"];

/// Deterministic xorshift64 PRNG for Xavier weight initialization (zero external rand dependency).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_f32_symmetric(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
        (u * 2.0) - 1.0
    }
}

/// Trainable parameter tensor with 1st and 2nd moment buffers for AdamW optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamTensor {
    pub w: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl ParamTensor {
    pub fn zeros(size: usize) -> Self {
        Self {
            w: vec![0.0; size],
            grad: vec![0.0; size],
            m: vec![0.0; size],
            v: vec![0.0; size],
        }
    }

    pub fn xavier(size: usize, fan_in: usize, fan_out: usize, rng: &mut XorShift64) -> Self {
        let scale = (6.0 / ((fan_in + fan_out) as f32)).sqrt();
        let mut w = Vec::with_capacity(size);
        for _ in 0..size {
            w.push(rng.next_f32_symmetric() * scale);
        }
        Self {
            w,
            grad: vec![0.0; size],
            m: vec![0.0; size],
            v: vec![0.0; size],
        }
    }

    pub fn zero_grad(&mut self) {
        for g in &mut self.grad {
            *g = 0.0;
        }
    }

    pub fn append_rows(&mut self, num_new_rows: usize, cols: usize, scale: f32, rng: &mut XorShift64) {
        let add_len = num_new_rows * cols;
        for _ in 0..add_len {
            self.w.push(rng.next_f32_symmetric() * scale);
        }
        self.grad.resize(self.w.len(), 0.0);
        self.m.resize(self.w.len(), 0.0);
        self.v.resize(self.w.len(), 0.0);
    }

    pub fn step_adamw(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, step_t: i32) {
        let b1_corr = 1.0 - beta1.powi(step_t);
        let b2_corr = 1.0 - beta2.powi(step_t);
        for i in 0..self.w.len() {
            let g = self.grad[i].clamp(-5.0, 5.0);
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * (g * g);
            let m_hat = self.m[i] / b1_corr;
            let v_hat = self.v[i] / b2_corr;
            self.w[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * self.w[i]);
            self.grad[i] = 0.0;
        }
    }
}

/// Differentiable Neural Logic MLU Cell with continuous T-norm gates (`AND`, `OR`, `IMPLIES`, `XOR`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DifferentiableLogicMluCell {
    pub cell_index: usize,
    pub w_a: ParamTensor,         // D x D
    pub b_a: ParamTensor,         // D
    pub w_b: ParamTensor,         // D x D
    pub b_b: ParamTensor,         // D
    pub gate_logits: ParamTensor, // D x 4
    pub w_o: ParamTensor,         // D x D
    pub b_o: ParamTensor,         // D
}

impl DifferentiableLogicMluCell {
    pub fn new(cell_index: usize, rng: &mut XorShift64) -> Self {
        let mut gate_logits = ParamTensor::zeros(D_MODEL * NUM_LOGIC_GATES);
        // Slight initial symmetry breaking across cells so each spoke cell specializes
        for d in 0..D_MODEL {
            let pref_gate = (cell_index + d) % NUM_LOGIC_GATES;
            gate_logits.w[d * NUM_LOGIC_GATES + pref_gate] = 0.35;
        }
        Self {
            cell_index,
            w_a: ParamTensor::xavier(D_MODEL * D_MODEL, D_MODEL, D_MODEL, rng),
            b_a: ParamTensor::zeros(D_MODEL),
            w_b: ParamTensor::xavier(D_MODEL * D_MODEL, D_MODEL, D_MODEL, rng),
            b_b: ParamTensor::zeros(D_MODEL),
            gate_logits,
            w_o: ParamTensor::xavier(D_MODEL * D_MODEL, D_MODEL, D_MODEL, rng),
            b_o: ParamTensor::zeros(D_MODEL),
        }
    }

    pub fn zero_grad(&mut self) {
        self.w_a.zero_grad();
        self.b_a.zero_grad();
        self.w_b.zero_grad();
        self.b_b.zero_grad();
        self.gate_logits.zero_grad();
        self.w_o.zero_grad();
        self.b_o.zero_grad();
    }

    pub fn step_adamw(&mut self, lr: f32, step_t: i32) {
        self.w_a.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, step_t);
        self.b_a.step_adamw(lr, 0.9, 0.999, 1e-8, 0.0, step_t);
        self.w_b.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, step_t);
        self.b_b.step_adamw(lr, 0.9, 0.999, 1e-8, 0.0, step_t);
        self.gate_logits.step_adamw(lr, 0.9, 0.999, 1e-8, 0.0, step_t);
        self.w_o.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, step_t);
        self.b_o.step_adamw(lr, 0.9, 0.999, 1e-8, 0.0, step_t);
    }

    pub fn gate_distribution_summary(&self) -> [f32; NUM_LOGIC_GATES] {
        let mut avg = [0.0f32; NUM_LOGIC_GATES];
        for d in 0..D_MODEL {
            let row = &self.gate_logits.w[d * NUM_LOGIC_GATES..(d + 1) * NUM_LOGIC_GATES];
            let probs = softmax_slice(row);
            for g in 0..NUM_LOGIC_GATES {
                avg[g] += probs[g];
            }
        }
        for g in 0..NUM_LOGIC_GATES {
            avg[g] = ((avg[g] / (D_MODEL as f32)) * 10000.0).round() / 10000.0;
        }
        avg
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-15.0, 15.0)).exp())
}

#[inline]
fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v - max_val).exp();
        exps.push(e);
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-12);
    for e in &mut exps {
        *e *= inv;
    }
    exps
}

#[inline]
fn matvec(mat: &[f32], rows: usize, cols: usize, vec: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let row_off = r * cols;
        let mut acc = bias.map_or(0.0, |b| b[r]);
        for c in 0..cols {
            acc += mat[row_off + c] * vec[c];
        }
        out[r] = acc;
    }
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralTrainReportDto {
    pub epochs: usize,
    pub sequences_trained: usize,
    pub total_tokens: usize,
    pub vocab_size: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub initial_perplexity: f32,
    pub final_perplexity: f32,
    pub optimizer_steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralGenerationDto {
    pub prompt: String,
    pub generated_text: String,
    pub generated_tokens: Vec<String>,
    pub mean_token_confidence: f32,
    pub active_spoke_weights: Vec<(String, f32)>,
    pub dominant_logic_gates: Vec<(String, String, f32)>,
}

/// Cached activations for a single timestep `t` during forward pass for analytical backpropagation.
struct StepCache {
    tok_id: usize,
    pos_idx: usize,
    x0: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_weights: Vec<f32>, // length t + 1
    attn_out: Vec<f32>,     // D
    g_tau: Vec<f32>,        // D
    h: Vec<f32>,            // D
    route_probs: Vec<f32>,  // K
    cell_a: Vec<Vec<f32>>,  // K x D
    cell_b: Vec<Vec<f32>>,  // K x D
    cell_gate_probs: Vec<Vec<[f32; NUM_LOGIC_GATES]>>, // K x D x 4
    cell_z: Vec<Vec<f32>>,  // K x D
    cell_m: Vec<Vec<f32>>,  // K x D
    o: Vec<f32>,            // D
    probs: Vec<f32>,        // V
}

/// Complete End-to-End Differentiable 360° Spider-Web Neural MLU Model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralSpiderWebMlu {
    pub vocab_to_id: HashMap<String, usize>,
    pub id_to_vocab: Vec<String>,
    pub rng: XorShift64,
    pub step_count: i32,

    pub token_emb: ParamTensor, // V x D
    pub pos_emb: ParamTensor,   // MAX_SEQ_LEN x D

    // Causal Liquid Self-Attention
    pub w_q: ParamTensor,   // D x D
    pub w_k: ParamTensor,   // D x D
    pub w_v: ParamTensor,   // D x D
    pub w_tau: ParamTensor, // D x D
    pub b_tau: ParamTensor, // D

    // 360° Spider-Web Softmax Router
    pub w_route: ParamTensor, // K x D
    pub b_route: ParamTensor, // K

    // 6 Radial Differentiable Logic MLU Cells
    pub logic_cells: Vec<DifferentiableLogicMluCell>,

    // Neural Language Head
    pub lm_head_w: ParamTensor, // V x D
    pub lm_head_b: ParamTensor, // V
}

impl NeuralSpiderWebMlu {
    pub fn new(seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);
        let mut vocab_to_id = HashMap::new();
        let mut id_to_vocab = Vec::new();
        for special in ["<pad>", "<bos>", "<eos>", "."] {
            vocab_to_id.insert(special.to_string(), id_to_vocab.len());
            id_to_vocab.push(special.to_string());
        }
        let v_init = id_to_vocab.len();
        let emb_scale = (1.0 / (D_MODEL as f32)).sqrt();

        let mut token_emb = ParamTensor::zeros(0);
        token_emb.append_rows(v_init, D_MODEL, emb_scale, &mut rng);

        let pos_emb = ParamTensor::xavier(MAX_SEQ_LEN * D_MODEL, MAX_SEQ_LEN, D_MODEL, &mut rng);

        let w_q = ParamTensor::xavier(D_MODEL * D_MODEL, D_MODEL, D_MODEL, &mut rng);
        let w_k = ParamTensor::xavier(D_MODEL * D_MODEL, D_MODEL, D_MODEL, &mut rng);
        let w_v = ParamTensor::xavier(D_MODEL * D_MODEL, D_MODEL, D_MODEL, &mut rng);
        let w_tau = ParamTensor::xavier(D_MODEL * D_MODEL, D_MODEL, D_MODEL, &mut rng);
        let b_tau = ParamTensor::zeros(D_MODEL);

        let w_route = ParamTensor::xavier(NUM_LOGIC_CELLS * D_MODEL, D_MODEL, NUM_LOGIC_CELLS, &mut rng);
        let b_route = ParamTensor::zeros(NUM_LOGIC_CELLS);

        let mut logic_cells = Vec::with_capacity(NUM_LOGIC_CELLS);
        for k in 0..NUM_LOGIC_CELLS {
            logic_cells.push(DifferentiableLogicMluCell::new(k, &mut rng));
        }

        let mut lm_head_w = ParamTensor::zeros(0);
        lm_head_w.append_rows(v_init, D_MODEL, emb_scale, &mut rng);
        let mut lm_head_b = ParamTensor::zeros(0);
        lm_head_b.append_rows(v_init, 1, 0.0, &mut rng);

        Self {
            vocab_to_id,
            id_to_vocab,
            rng,
            step_count: 0,
            token_emb,
            pos_emb,
            w_q,
            w_k,
            w_v,
            w_tau,
            b_tau,
            w_route,
            b_route,
            logic_cells,
            lm_head_w,
            lm_head_b,
        }
    }

    /// Ensure token exists in dynamic neural vocabulary, expanding `token_emb` and `lm_head` if new.
    pub fn ensure_token(&mut self, tok: &str) -> usize {
        if let Some(&id) = self.vocab_to_id.get(tok) {
            return id;
        }
        let new_id = self.id_to_vocab.len();
        self.vocab_to_id.insert(tok.to_string(), new_id);
        self.id_to_vocab.push(tok.to_string());
        let scale = (1.0 / (D_MODEL as f32)).sqrt();
        self.token_emb.append_rows(1, D_MODEL, scale, &mut self.rng);
        self.lm_head_w.append_rows(1, D_MODEL, scale, &mut self.rng);
        self.lm_head_b.append_rows(1, 1, 0.0, &mut self.rng);
        new_id
    }

    /// Pure character-state tokenizer (words + sentence punctuation `.` and `?`), zero regexes.
    pub fn tokenize_text(text: &str) -> Vec<Vec<String>> {
        let mut sentences: Vec<Vec<String>> = Vec::new();
        let mut curr_sent: Vec<String> = Vec::new();
        let mut buf = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                buf.push(ch.to_ascii_lowercase());
            } else {
                if !buf.is_empty() {
                    curr_sent.push(buf.clone());
                    buf.clear();
                }
                if ch == '?' {
                    curr_sent.push("?".to_string());
                } else if ch == '.' || ch == '!' || ch == '\n' {
                    if !curr_sent.is_empty() {
                        curr_sent.push(".".to_string());
                        sentences.push(curr_sent.clone());
                        curr_sent.clear();
                    }
                }
            }
        }
        if !buf.is_empty() {
            curr_sent.push(buf);
        }
        if !curr_sent.is_empty() {
            if curr_sent.last().map(String::as_str) != Some(".")
                && curr_sent.last().map(String::as_str) != Some("?")
            {
                curr_sent.push(".".to_string());
            }
            sentences.push(curr_sent);
        }
        sentences
    }

    /// Forward pass over a sequence of token IDs `seq`, returning cached activations for backprop.
    fn forward_sequence(&self, seq: &[usize]) -> Vec<StepCache> {
        let len = seq.len().min(MAX_SEQ_LEN);
        let v_size = self.id_to_vocab.len();
        let mut caches: Vec<StepCache> = Vec::with_capacity(len);
        let scale_qk = 1.0 / (D_MODEL as f32).sqrt();

        let mut all_k: Vec<Vec<f32>> = Vec::with_capacity(len);
        let mut all_v: Vec<Vec<f32>> = Vec::with_capacity(len);
        let mut liquid_state = vec![0.0f32; D_MODEL];

        for t in 0..len {
            let tok_id = seq[t];
            let pos_idx = t;
            let mut x0 = vec![0.0f32; D_MODEL];
            let emb_off = tok_id * D_MODEL;
            let pos_off = pos_idx * D_MODEL;
            let prev1_off = if t >= 1 { Some(seq[t - 1] * D_MODEL) } else { None };
            for d in 0..D_MODEL {
                let mut inp = self.token_emb.w[emb_off + d] + self.pos_emb.w[pos_off + d];
                if let Some(p1) = prev1_off {
                    inp += 0.50 * self.token_emb.w[p1 + d];
                }
                liquid_state[d] = 0.78 * liquid_state[d] + inp;
                x0[d] = liquid_state[d];
            }

            let q = matvec(&self.w_q.w, D_MODEL, D_MODEL, &x0, None);
            let k_vec = matvec(&self.w_k.w, D_MODEL, D_MODEL, &x0, None);
            let v_vec = matvec(&self.w_v.w, D_MODEL, D_MODEL, &x0, None);
            all_k.push(k_vec.clone());
            all_v.push(v_vec.clone());

            // Causal attention scores over j <= t
            let mut scores = Vec::with_capacity(t + 1);
            for j in 0..=t {
                let mut dot = 0.0f32;
                for d in 0..D_MODEL {
                    dot += q[d] * all_k[j][d];
                }
                scores.push(dot * scale_qk);
            }
            let attn_weights = softmax_slice(&scores);
            let mut attn_out = vec![0.0f32; D_MODEL];
            for j in 0..=t {
                let alpha = attn_weights[j];
                for d in 0..D_MODEL {
                    attn_out[d] += alpha * all_v[j][d];
                }
            }

            // Liquid Time-Constant Gate g_t = sigmoid(W_tau * x0 + b_tau)
            let tau_pre = matvec(&self.w_tau.w, D_MODEL, D_MODEL, &x0, Some(&self.b_tau.w));
            let mut g_tau = vec![0.0f32; D_MODEL];
            let mut h = vec![0.0f32; D_MODEL];
            for d in 0..D_MODEL {
                g_tau[d] = sigmoid(tau_pre[d]);
                h[d] = x0[d] + g_tau[d] * attn_out[d];
            }

            // 360° Softmax Router over K = 6 Differentiable Logic MLU Cells
            let route_logits = matvec(
                &self.w_route.w,
                NUM_LOGIC_CELLS,
                D_MODEL,
                &h,
                Some(&self.b_route.w),
            );
            let route_probs = softmax_slice(&route_logits);

            let mut cell_a = Vec::with_capacity(NUM_LOGIC_CELLS);
            let mut cell_b = Vec::with_capacity(NUM_LOGIC_CELLS);
            let mut cell_gate_probs = Vec::with_capacity(NUM_LOGIC_CELLS);
            let mut cell_z = Vec::with_capacity(NUM_LOGIC_CELLS);
            let mut cell_m = Vec::with_capacity(NUM_LOGIC_CELLS);

            let mut o = h.clone();

            for (c_idx, cell) in self.logic_cells.iter().enumerate() {
                let pre_a = matvec(&cell.w_a.w, D_MODEL, D_MODEL, &h, Some(&cell.b_a.w));
                let pre_b = matvec(&cell.w_b.w, D_MODEL, D_MODEL, &h, Some(&cell.b_b.w));
                let mut a_vec = vec![0.0f32; D_MODEL];
                let mut b_vec = vec![0.0f32; D_MODEL];
                let mut g_probs_d = Vec::with_capacity(D_MODEL);
                let mut z_vec = vec![0.0f32; D_MODEL];

                for d in 0..D_MODEL {
                    let a = sigmoid(pre_a[d]);
                    let b = sigmoid(pre_b[d]);
                    a_vec[d] = a;
                    b_vec[d] = b;

                    let g_slice =
                        &cell.gate_logits.w[d * NUM_LOGIC_GATES..(d + 1) * NUM_LOGIC_GATES];
                    let gp = softmax_slice(g_slice);
                    let gp_arr = [gp[0], gp[1], gp[2], gp[3]];
                    g_probs_d.push(gp_arr);

                    let l_and = a * b;
                    let l_or = a + b - a * b;
                    let l_imp = 1.0 - a + a * b;
                    let l_xor = a + b - 2.0 * a * b;

                    z_vec[d] = gp_arr[0] * l_and
                        + gp_arr[1] * l_or
                        + gp_arr[2] * l_imp
                        + gp_arr[3] * l_xor;
                }

                let m_vec = matvec(&cell.w_o.w, D_MODEL, D_MODEL, &z_vec, Some(&cell.b_o.w));
                let r_k = route_probs[c_idx];
                for d in 0..D_MODEL {
                    o[d] += r_k * m_vec[d];
                }

                cell_a.push(a_vec);
                cell_b.push(b_vec);
                cell_gate_probs.push(g_probs_d);
                cell_z.push(z_vec);
                cell_m.push(m_vec);
            }

            let logits = matvec(
                &self.lm_head_w.w,
                v_size,
                D_MODEL,
                &o,
                Some(&self.lm_head_b.w),
            );
            let probs = softmax_slice(&logits);

            caches.push(StepCache {
                tok_id,
                pos_idx,
                x0,
                q,
                k: k_vec,
                v: v_vec,
                attn_weights,
                attn_out,
                g_tau,
                h,
                route_probs,
                cell_a,
                cell_b,
                cell_gate_probs,
                cell_z,
                cell_m,
                o,
                probs,
            });
        }

        caches
    }

    /// Analytical Backpropagation + AdamW update over a single sequence of token IDs `seq`.
    /// Returns the average cross-entropy loss across the sequence.
    pub fn train_sequence_step(&mut self, seq: &[usize], lr: f32) -> f32 {
        let len = seq.len().min(MAX_SEQ_LEN);
        if len < 2 {
            return 0.0;
        }
        let caches = self.forward_sequence(&seq[..len]);
        let v_size = self.id_to_vocab.len();
        let scale_qk = 1.0 / (D_MODEL as f32).sqrt();
        let inv_steps = 1.0 / ((len - 1) as f32);

        let mut total_loss = 0.0f32;
        let mut d_all_k = vec![vec![0.0f32; D_MODEL]; len];
        let mut d_all_v = vec![vec![0.0f32; D_MODEL]; len];
        let mut d_x0_list = vec![vec![0.0f32; D_MODEL]; len];
        let mut d_q_list = vec![vec![0.0f32; D_MODEL]; len];

        for t in 0..(len - 1) {
            let target_id = seq[t + 1];
            let cache = &caches[t];
            let p_target = cache.probs[target_id].max(1e-12);
            total_loss += -p_target.ln();

            // 1. dL / d_logits = (probs - one_hot(target)) * inv_steps
            let mut d_o = vec![0.0f32; D_MODEL];
            for v in 0..v_size {
                let mut dl = cache.probs[v];
                if v == target_id {
                    dl -= 1.0;
                }
                dl *= inv_steps;
                if dl.abs() < 1e-9 {
                    continue;
                }
                self.lm_head_b.grad[v] += dl;
                let row_off = v * D_MODEL;
                for d in 0..D_MODEL {
                    self.lm_head_w.grad[row_off + d] += dl * cache.o[d];
                    d_o[d] += dl * self.lm_head_w.w[row_off + d];
                }
            }

            // 2. Backprop through 360° Router and 6 Differentiable Logic MLU Cells
            let mut d_h = d_o.clone();
            let mut d_route_probs = [0.0f32; NUM_LOGIC_CELLS];

            for c_idx in 0..NUM_LOGIC_CELLS {
                let r_k = cache.route_probs[c_idx];
                let m_vec = &cache.cell_m[c_idx];
                let z_vec = &cache.cell_z[c_idx];
                let a_vec = &cache.cell_a[c_idx];
                let b_vec = &cache.cell_b[c_idx];
                let gp_vec = &cache.cell_gate_probs[c_idx];

                let mut d_m = vec![0.0f32; D_MODEL];
                let mut dot_do_m = 0.0f32;
                for d in 0..D_MODEL {
                    dot_do_m += d_o[d] * m_vec[d];
                    d_m[d] = d_o[d] * r_k;
                }
                d_route_probs[c_idx] = dot_do_m;

                let cell = &mut self.logic_cells[c_idx];
                let mut d_z = vec![0.0f32; D_MODEL];
                for r in 0..D_MODEL {
                    let dm_r = d_m[r];
                    cell.b_o.grad[r] += dm_r;
                    let row_off = r * D_MODEL;
                    for c in 0..D_MODEL {
                        cell.w_o.grad[row_off + c] += dm_r * z_vec[c];
                        d_z[c] += dm_r * cell.w_o.w[row_off + c];
                    }
                }

                // Backprop through Differentiable Logic Gates (AND, OR, IMPLIES, XOR)
                let mut d_pre_a = vec![0.0f32; D_MODEL];
                let mut d_pre_b = vec![0.0f32; D_MODEL];

                for d in 0..D_MODEL {
                    let dz = d_z[d];
                    let a = a_vec[d];
                    let b = b_vec[d];
                    let gp = gp_vec[d];

                    let l_vals = [
                        a * b,
                        a + b - a * b,
                        1.0 - a + a * b,
                        a + b - 2.0 * a * b,
                    ];

                    // Softmax jacobian for gate_logits[d, 0..4]
                    let mut dot_gp_l = 0.0f32;
                    for g in 0..NUM_LOGIC_GATES {
                        dot_gp_l += gp[g] * l_vals[g];
                    }
                    for g in 0..NUM_LOGIC_GATES {
                        let d_u = dz * gp[g] * (l_vals[g] - dot_gp_l);
                        cell.gate_logits.grad[d * NUM_LOGIC_GATES + g] += d_u;
                    }

                    // Gradients wrt operands a and b
                    let dl_da = gp[0] * b
                        + gp[1] * (1.0 - b)
                        + gp[2] * (b - 1.0)
                        + gp[3] * (1.0 - 2.0 * b);
                    let dl_db = gp[0] * a
                        + gp[1] * (1.0 - a)
                        + gp[2] * a
                        + gp[3] * (1.0 - 2.0 * a);

                    d_pre_a[d] = dz * dl_da * a * (1.0 - a);
                    d_pre_b[d] = dz * dl_db * b * (1.0 - b);
                }

                for r in 0..D_MODEL {
                    let da_r = d_pre_a[r];
                    let db_r = d_pre_b[r];
                    cell.b_a.grad[r] += da_r;
                    cell.b_b.grad[r] += db_r;
                    let row_off = r * D_MODEL;
                    for c in 0..D_MODEL {
                        cell.w_a.grad[row_off + c] += da_r * cache.h[c];
                        cell.w_b.grad[row_off + c] += db_r * cache.h[c];
                        d_h[c] += da_r * cell.w_a.w[row_off + c] + db_r * cell.w_b.w[row_off + c];
                    }
                }
            }

            // Router softmax jacobian
            let mut dot_r = 0.0f32;
            for k in 0..NUM_LOGIC_CELLS {
                dot_r += cache.route_probs[k] * d_route_probs[k];
            }
            for k in 0..NUM_LOGIC_CELLS {
                let d_rl = cache.route_probs[k] * (d_route_probs[k] - dot_r);
                self.b_route.grad[k] += d_rl;
                let row_off = k * D_MODEL;
                for d in 0..D_MODEL {
                    self.w_route.grad[row_off + d] += d_rl * cache.h[d];
                    d_h[d] += d_rl * self.w_route.w[row_off + d];
                }
            }

            // 3. Backprop through Liquid Time-Constant Gate & Causal Self-Attention
            let mut d_x0 = d_h.clone();
            let mut d_attn_out = vec![0.0f32; D_MODEL];
            let mut d_tau_pre = vec![0.0f32; D_MODEL];

            for d in 0..D_MODEL {
                d_attn_out[d] = d_h[d] * cache.g_tau[d];
                let dg = d_h[d] * cache.attn_out[d];
                d_tau_pre[d] = dg * cache.g_tau[d] * (1.0 - cache.g_tau[d]);
            }

            for r in 0..D_MODEL {
                let dt_r = d_tau_pre[r];
                self.b_tau.grad[r] += dt_r;
                let row_off = r * D_MODEL;
                for c in 0..D_MODEL {
                    self.w_tau.grad[row_off + c] += dt_r * cache.x0[c];
                    d_x0[c] += dt_r * self.w_tau.w[row_off + c];
                }
            }

            // Backprop through attn_out = sum_{j<=t} alpha_{t,j} * v_j
            let mut d_alpha = vec![0.0f32; t + 1];
            for j in 0..=t {
                let alpha = cache.attn_weights[j];
                let mut dot_da_v = 0.0f32;
                for d in 0..D_MODEL {
                    d_all_v[j][d] += alpha * d_attn_out[d];
                    dot_da_v += d_attn_out[d] * caches[j].v[d];
                }
                d_alpha[j] = dot_da_v;
            }

            let mut dot_alpha = 0.0f32;
            for j in 0..=t {
                dot_alpha += cache.attn_weights[j] * d_alpha[j];
            }
            for j in 0..=t {
                let d_score = cache.attn_weights[j] * (d_alpha[j] - dot_alpha) * scale_qk;
                for d in 0..D_MODEL {
                    d_q_list[t][d] += d_score * caches[j].k[d];
                    d_all_k[j][d] += d_score * cache.q[d];
                }
            }

            for d in 0..D_MODEL {
                d_x0_list[t][d] += d_x0[d];
            }
        }

        // Accumulate Q, K, V gradients into W_Q, W_K, W_V and backpropagate through time (BPTT) into token_emb / pos_emb
        let mut d_liquid_next = vec![0.0f32; D_MODEL];
        for t in (0..len).rev() {
            let cache = &caches[t];
            let mut d_x0 = d_x0_list[t].clone();
            for r in 0..D_MODEL {
                let dq_r = d_q_list[t][r];
                let dk_r = d_all_k[t][r];
                let dv_r = d_all_v[t][r];
                let row_off = r * D_MODEL;
                for c in 0..D_MODEL {
                    let x_c = cache.x0[c];
                    self.w_q.grad[row_off + c] += dq_r * x_c;
                    self.w_k.grad[row_off + c] += dk_r * x_c;
                    self.w_v.grad[row_off + c] += dv_r * x_c;
                    d_x0[c] += dq_r * self.w_q.w[row_off + c]
                        + dk_r * self.w_k.w[row_off + c]
                        + dv_r * self.w_v.w[row_off + c];
                }
            }

            for d in 0..D_MODEL {
                d_x0[d] += d_liquid_next[d];
                d_liquid_next[d] = 0.78 * d_x0[d];
            }

            let emb_off = cache.tok_id * D_MODEL;
            let pos_off = cache.pos_idx * D_MODEL;
            let prev1_off = if t >= 1 { Some(seq[t - 1] * D_MODEL) } else { None };
            for d in 0..D_MODEL {
                self.token_emb.grad[emb_off + d] += d_x0[d];
                self.pos_emb.grad[pos_off + d] += d_x0[d];
                if let Some(p1) = prev1_off {
                    self.token_emb.grad[p1 + d] += 0.50 * d_x0[d];
                }
            }
        }

        // AdamW parameter update step
        self.step_count += 1;
        let st = self.step_count;
        self.token_emb.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, st);
        self.pos_emb.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, st);
        self.w_q.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, st);
        self.w_k.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, st);
        self.w_v.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, st);
        self.w_tau.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, st);
        self.b_tau.step_adamw(lr, 0.9, 0.999, 1e-8, 0.0, st);
        self.w_route.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, st);
        self.b_route.step_adamw(lr, 0.9, 0.999, 1e-8, 0.0, st);
        for cell in &mut self.logic_cells {
            cell.step_adamw(lr, st);
        }
        self.lm_head_w.step_adamw(lr, 0.9, 0.999, 1e-8, 1e-4, st);
        self.lm_head_b.step_adamw(lr, 0.9, 0.999, 1e-8, 0.0, st);

        total_loss * inv_steps
    }

    /// Train the Neural 360° Spider-Web MLU end-to-end on raw text across `epochs`.
    pub fn train_text_corpus(&mut self, text: &str, epochs: usize, lr: f32) -> NeuralTrainReportDto {
        let raw_sents = Self::tokenize_text(text);
        if raw_sents.is_empty() {
            return NeuralTrainReportDto {
                epochs: 0,
                sequences_trained: 0,
                total_tokens: 0,
                vocab_size: self.id_to_vocab.len(),
                initial_loss: 0.0,
                final_loss: 0.0,
                initial_perplexity: 1.0,
                final_perplexity: 1.0,
                optimizer_steps: self.step_count as usize,
            };
        }

        let bos_id = self.ensure_token("<bos>");
        let eos_id = self.ensure_token("<eos>");

        let mut id_sents: Vec<Vec<usize>> = Vec::with_capacity(raw_sents.len());
        let mut total_tokens = 0usize;
        for s in &raw_sents {
            let mut ids = Vec::with_capacity(s.len());
            for tok in s {
                ids.push(self.ensure_token(tok));
                total_tokens += 1;
            }
            id_sents.push(ids);
        }

        // Build balanced training sequences
        let mut train_seqs: Vec<Vec<usize>> = Vec::new();
        for i in 0..id_sents.len() {
            let mut seq1 = vec![bos_id];
            seq1.extend_from_slice(&id_sents[i]);
            seq1.push(eos_id);
            if seq1.len() <= MAX_SEQ_LEN {
                train_seqs.push(seq1);
            }
        }

        let ep_count = epochs.max(1);
        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for ep in 0..ep_count {
            let mut ep_loss = 0.0f32;
            let curr_lr = lr * (0.98f32).powi(ep as i32);
            for seq in &train_seqs {
                ep_loss += self.train_sequence_step(seq, curr_lr);
            }
            let avg_ep_loss = ep_loss / (train_seqs.len().max(1) as f32);
            if ep == 0 {
                initial_loss = avg_ep_loss;
            }
            final_loss = avg_ep_loss;
        }

        NeuralTrainReportDto {
            epochs: ep_count,
            sequences_trained: train_seqs.len() * ep_count,
            total_tokens,
            vocab_size: self.id_to_vocab.len(),
            initial_loss: (initial_loss * 10000.0).round() / 10000.0,
            final_loss: (final_loss * 10000.0).round() / 10000.0,
            initial_perplexity: ((initial_loss.exp()) * 100.0).round() / 100.0,
            final_perplexity: ((final_loss.exp()) * 100.0).round() / 100.0,
            optimizer_steps: self.step_count as usize,
        }
    }

    /// Generate a response token-by-token purely from the Neural 360° Spider-Web MLU weights.
    pub fn generate_response(&self, prompt: &str, max_new_tokens: usize) -> NeuralGenerationDto {
        let sents = Self::tokenize_text(prompt);
        let mut prompt_toks: Vec<String> = Vec::new();
        for (idx, s) in sents.iter().enumerate() {
            for tok in s {
                // Keep trailing '.' only if prompt explicitly had it or if question mark '?'
                if tok == "." && idx + 1 == sents.len() && !prompt.trim_end().ends_with('.') {
                    continue;
                }
                prompt_toks.push(tok.clone());
            }
        }

        let bos_id = *self.vocab_to_id.get("<bos>").unwrap_or(&1);
        let eos_id = *self.vocab_to_id.get("<eos>").unwrap_or(&2);
        let pad_id = *self.vocab_to_id.get("<pad>").unwrap_or(&0);

        let mut seq: Vec<usize> = vec![bos_id];
        for tok in &prompt_toks {
            if let Some(&id) = self.vocab_to_id.get(tok) {
                seq.push(id);
            }
        }

        let limit = max_new_tokens.clamp(1, 32);
        let mut generated_tokens: Vec<String> = Vec::new();
        let mut conf_sum = 0.0f32;
        let mut route_accum = [0.0f32; NUM_LOGIC_CELLS];
        let mut steps_done = 0usize;

        for _ in 0..limit {
            if seq.len() >= MAX_SEQ_LEN {
                break;
            }
            let caches = self.forward_sequence(&seq);
            let Some(last) = caches.last() else {
                break;
            };
            for k in 0..NUM_LOGIC_CELLS {
                route_accum[k] += last.route_probs[k];
            }
            steps_done += 1;

            let mut best_id = eos_id;
            let mut best_prob = -1.0f32;
            for (vid, &p) in last.probs.iter().enumerate() {
                if vid == pad_id || vid == bos_id {
                    continue;
                }
                // Mild repetition penalty against immediately repeating the last 1-2 tokens
                let tok_str = &self.id_to_vocab[vid];
                let penalty = if generated_tokens.iter().rev().take(2).any(|t| t == tok_str) && tok_str != "." {
                    0.35
                } else {
                    1.0
                };
                let adj_p = p * penalty;
                if adj_p > best_prob {
                    best_prob = adj_p;
                    best_id = vid;
                }
            }

            if best_id == eos_id {
                break;
            }
            let next_word = self.id_to_vocab[best_id].clone();
            conf_sum += last.probs[best_id];
            generated_tokens.push(next_word.clone());
            seq.push(best_id);
            if next_word == "." && generated_tokens.len() >= 2 {
                break;
            }
        }

        // Format generated token stream into clean English spacing
        let mut out_str = String::new();
        for tok in &generated_tokens {
            if tok == "." || tok == "?" {
                out_str.push_str(tok);
            } else {
                if !out_str.is_empty() {
                    out_str.push(' ');
                }
                out_str.push_str(tok);
            }
        }
        if let Some(first) = out_str.get_mut(0..1) {
            first.make_ascii_uppercase();
        }

        let mut active_spoke_weights = Vec::with_capacity(NUM_LOGIC_CELLS);
        let denom = (steps_done.max(1)) as f32;
        for k in 0..NUM_LOGIC_CELLS {
            let w = ((route_accum[k] / denom) * 10000.0).round() / 10000.0;
            active_spoke_weights.push((SPOKE_CELL_NAMES[k].to_string(), w));
        }
        active_spoke_weights.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut dominant_logic_gates = Vec::with_capacity(NUM_LOGIC_CELLS);
        for k in 0..NUM_LOGIC_CELLS {
            let dist = self.logic_cells[k].gate_distribution_summary();
            let mut best_g = 0;
            for g in 1..NUM_LOGIC_GATES {
                if dist[g] > dist[best_g] {
                    best_g = g;
                }
            }
            dominant_logic_gates.push((
                SPOKE_CELL_NAMES[k].to_string(),
                LOGIC_GATE_NAMES[best_g].to_string(),
                dist[best_g],
            ));
        }

        let mean_conf = if generated_tokens.is_empty() {
            0.0
        } else {
            ((conf_sum / (generated_tokens.len() as f32)) * 10000.0).round() / 10000.0
        };

        NeuralGenerationDto {
            prompt: prompt.to_string(),
            generated_text: out_str,
            generated_tokens,
            mean_token_confidence: mean_conf,
            active_spoke_weights,
            dominant_logic_gates,
        }
    }
}
