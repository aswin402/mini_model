//! High-performance Rust acceleration core for LITTLE's cognitive graph operations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

pub mod mlu_arena;
pub mod neural_mlu;
use mlu_arena::SpiderWebArena;
use neural_mlu::NeuralSpiderWebMlu;

// ---------------------------------------------------------------------------
// Pure Rust Graph Algorithms
// ---------------------------------------------------------------------------

/// Compute all transitive ancestors reachable from start_node via outgoing edges.
pub fn find_ancestors(adjacency: &HashMap<String, Vec<String>>, start_node: &str) -> Vec<String> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = Vec::new();

    if let Some(neighbors) = adjacency.get(start_node) {
        for n in neighbors {
            stack.push(n.clone());
        }
    }

    while let Some(curr) = stack.pop() {
        if visited.insert(curr.clone()) {
            if let Some(neighbors) = adjacency.get(&curr) {
                for n in neighbors {
                    if !visited.contains(n) {
                        stack.push(n.clone());
                    }
                }
            }
        }
    }

    visited.into_iter().collect()
}

/// Check if a directed path exists from start_node to target_node (Cycle detection / I_DAG).
pub fn check_cycle(
    adjacency: &HashMap<String, Vec<String>>,
    start_node: &str,
    target_node: &str,
) -> bool {
    let mut visited: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = vec![start_node.to_string()];

    while let Some(curr) = stack.pop() {
        if curr == target_node {
            return true;
        }
        if visited.insert(curr.clone()) {
            if let Some(neighbors) = adjacency.get(&curr) {
                for n in neighbors {
                    if n == target_node {
                        return true;
                    }
                    if !visited.contains(n) {
                        stack.push(n.clone());
                    }
                }
            }
        }
    }
    false
}

/// Kahn's algorithm for Poset topological ordering.
pub fn topological_sort(
    nodes: &[String],
    edges: &[(String, String)],
) -> Result<Vec<String>, String> {
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

    for node in nodes {
        in_degree.insert(node.as_str(), 0);
        adj.insert(node.as_str(), Vec::new());
    }

    for (src, dst) in edges {
        adj.entry(src.as_str()).or_default().push(dst.as_str());
        *in_degree.entry(dst.as_str()).or_default() += 1;
    }

    let mut queue: VecDeque<&str> = VecDeque::new();
    for (&node, &deg) in &in_degree {
        if deg == 0 {
            queue.push_back(node);
        }
    }

    let mut result: Vec<String> = Vec::with_capacity(nodes.len());
    while let Some(node) = queue.pop_front() {
        result.push(node.to_string());
        if let Some(neighbors) = adj.get(node) {
            for &n in neighbors {
                let deg = in_degree.get_mut(n).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(n);
                }
            }
        }
    }

    if result.len() == nodes.len() {
        Ok(result)
    } else {
        Err("Cycle detected: graph is not a Directed Acyclic Graph (DAG)".to_string())
    }
}

/// Bidirectional Meeting-in-the-Middle frontier collision search.
pub fn bidirectional_search(
    forward_adj: &HashMap<String, Vec<String>>,
    backward_adj: &HashMap<String, Vec<String>>,
    start: &str,
    target: &str,
    max_depth: usize,
) -> Option<Vec<String>> {
    if start == target {
        return Some(vec![start.to_string()]);
    }

    let mut visited_fwd: HashMap<String, Vec<String>> = HashMap::new();
    let mut visited_bwd: HashMap<String, Vec<String>> = HashMap::new();

    let mut q_fwd: VecDeque<String> = VecDeque::new();
    let mut q_bwd: VecDeque<String> = VecDeque::new();

    visited_fwd.insert(start.to_string(), vec![start.to_string()]);
    visited_bwd.insert(target.to_string(), vec![target.to_string()]);

    q_fwd.push_back(start.to_string());
    q_bwd.push_back(target.to_string());

    let mut depth = 0;
    while (!q_fwd.is_empty() || !q_bwd.is_empty()) && depth < max_depth {
        depth += 1;

        // Step Forward
        let fwd_size = q_fwd.len();
        for _ in 0..fwd_size {
            if let Some(curr) = q_fwd.pop_front() {
                let path = visited_fwd.get(&curr).cloned().unwrap();
                if let Some(neighbors) = forward_adj.get(&curr) {
                    for n in neighbors {
                        if visited_bwd.contains_key(n) {
                            let mut full_path = path.clone();
                            let bwd_path = visited_bwd.get(n).unwrap();
                            let mut bwd_rev = bwd_path.clone();
                            bwd_rev.reverse();
                            full_path.extend(bwd_rev);
                            return Some(full_path);
                        }
                        if !visited_fwd.contains_key(n) {
                            let mut next_path = path.clone();
                            next_path.push(n.clone());
                            visited_fwd.insert(n.clone(), next_path);
                            q_fwd.push_back(n.clone());
                        }
                    }
                }
            }
        }

        // Step Backward
        let bwd_size = q_bwd.len();
        for _ in 0..bwd_size {
            if let Some(curr) = q_bwd.pop_front() {
                let path = visited_bwd.get(&curr).cloned().unwrap();
                if let Some(neighbors) = backward_adj.get(&curr) {
                    for n in neighbors {
                        if visited_fwd.contains_key(n) {
                            let fwd_path = visited_fwd.get(n).unwrap();
                            let mut full_path = fwd_path.clone();
                            let mut bwd_rev = path.clone();
                            bwd_rev.reverse();
                            full_path.extend(bwd_rev);
                            return Some(full_path);
                        }
                        if !visited_bwd.contains_key(n) {
                            let mut next_path = path.clone();
                            next_path.push(n.clone());
                            visited_bwd.insert(n.clone(), next_path);
                            q_bwd.push_back(n.clone());
                        }
                    }
                }
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// C-ABI FFI Exports (JSON serialization for zero-boilerplate FFI)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AncestorQuery {
    adjacency: HashMap<String, Vec<String>>,
    start_node: String,
}

#[derive(Serialize)]
struct AncestorResponse {
    ancestors: Vec<String>,
}

#[derive(Deserialize)]
struct CycleQuery {
    adjacency: HashMap<String, Vec<String>>,
    start_node: String,
    target_node: String,
}

#[derive(Deserialize)]
struct TopoQuery {
    nodes: Vec<String>,
    edges: Vec<(String, String)>,
}

#[derive(Serialize)]
struct TopoResponse {
    success: bool,
    order: Vec<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct BiSearchQuery {
    forward_adj: HashMap<String, Vec<String>>,
    backward_adj: HashMap<String, Vec<String>>,
    start: String,
    target: String,
    max_depth: usize,
}

#[derive(Serialize)]
struct BiSearchResponse {
    found: bool,
    path: Option<Vec<String>>,
}

fn to_c_string<T: Serialize>(val: &T) -> *mut c_char {
    let json = serde_json::to_string(val).unwrap_or_else(|_| "{}".to_string());
    CString::new(json).unwrap().into_raw()
}

fn from_c_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr).to_str().ok() }
}

#[no_mangle]
pub extern "C" fn little_core_find_ancestors(json_input: *const c_char) -> *mut c_char {
    let raw = match from_c_str(json_input) {
        Some(s) => s,
        None => return to_c_string(&AncestorResponse { ancestors: vec![] }),
    };

    let query: AncestorQuery = match serde_json::from_str(raw) {
        Ok(q) => q,
        Err(_) => return to_c_string(&AncestorResponse { ancestors: vec![] }),
    };

    let ans = find_ancestors(&query.adjacency, &query.start_node);
    to_c_string(&AncestorResponse { ancestors: ans })
}

#[no_mangle]
pub extern "C" fn little_core_check_cycle(json_input: *const c_char) -> i32 {
    let raw = match from_c_str(json_input) {
        Some(s) => s,
        None => return 0,
    };

    let query: CycleQuery = match serde_json::from_str(raw) {
        Ok(q) => q,
        Err(_) => return 0,
    };

    if check_cycle(&query.adjacency, &query.start_node, &query.target_node) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn little_core_toposort(json_input: *const c_char) -> *mut c_char {
    let raw = match from_c_str(json_input) {
        Some(s) => s,
        None => {
            return to_c_string(&TopoResponse {
                success: false,
                order: vec![],
                error: Some("Null pointer".to_string()),
            })
        }
    };

    let query: TopoQuery = match serde_json::from_str(raw) {
        Ok(q) => q,
        Err(e) => {
            return to_c_string(&TopoResponse {
                success: false,
                order: vec![],
                error: Some(e.to_string()),
            })
        }
    };

    match topological_sort(&query.nodes, &query.edges) {
        Ok(order) => to_c_string(&TopoResponse {
            success: true,
            order,
            error: None,
        }),
        Err(err) => to_c_string(&TopoResponse {
            success: false,
            order: vec![],
            error: Some(err),
        }),
    }
}

#[no_mangle]
pub extern "C" fn little_core_bidirectional_search(json_input: *const c_char) -> *mut c_char {
    let raw = match from_c_str(json_input) {
        Some(s) => s,
        None => {
            return to_c_string(&BiSearchResponse {
                found: false,
                path: None,
            })
        }
    };

    let query: BiSearchQuery = match serde_json::from_str(raw) {
        Ok(q) => q,
        Err(_) => {
            return to_c_string(&BiSearchResponse {
                found: false,
                path: None,
            })
        }
    };

    match bidirectional_search(
        &query.forward_adj,
        &query.backward_adj,
        &query.start,
        &query.target,
        query.max_depth,
    ) {
        Some(p) => to_c_string(&BiSearchResponse {
            found: true,
            path: Some(p),
        }),
        None => to_c_string(&BiSearchResponse {
            found: false,
            path: None,
        }),
    }
}

#[no_mangle]
pub extern "C" fn little_core_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// TurboQuant Vector Compression & Sub-Millisecond Search FFI
// ---------------------------------------------------------------------------

pub mod turbo;
use turbo::TurboIndexRust;

#[no_mangle]
pub extern "C" fn little_core_turbo_index_create(
    dimension: usize,
    signs_ptr: *const f32,
    signs_len: usize,
) -> *mut TurboIndexRust {
    if signs_ptr.is_null() || dimension != 512 || signs_len != 512 {
        return std::ptr::null_mut();
    }
    // SAFETY: The caller supplies signs_len readable floats; null and supported
    // dimensions have been checked before constructing the slice.
    let signs = unsafe { std::slice::from_raw_parts(signs_ptr, signs_len) };
    let index = match TurboIndexRust::new(dimension, signs) {
        Ok(index) => Box::new(index),
        Err(_) => return std::ptr::null_mut(),
    };
    Box::into_raw(index)
}

#[no_mangle]
pub extern "C" fn little_core_turbo_index_destroy(index_ptr: *mut TurboIndexRust) {
    if !index_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(index_ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn little_core_turbo_index_add(
    index_ptr: *mut TurboIndexRust,
    id_cstr: *const c_char,
    vector_ptr: *const f32,
    vector_len: usize,
    metadata_json_cstr: *const c_char,
) -> i32 {
    if index_ptr.is_null() || vector_ptr.is_null() {
        return 0;
    }
    let id_str = match from_c_str(id_cstr) {
        Some(s) => s,
        None => return 0,
    };
    let meta_str = match from_c_str(metadata_json_cstr) {
        Some(s) => s,
        None => "{}",
    };

    let index = unsafe { &mut *index_ptr };
    if vector_len != index.dimension {
        return 0;
    }
    let vector = unsafe { std::slice::from_raw_parts(vector_ptr, vector_len) };

    if vector.iter().any(|value| !value.is_finite()) {
        return 0;
    }
    index.add(id_str, vector, meta_str);
    1
}

#[no_mangle]
pub extern "C" fn little_core_turbo_index_remove(
    index_ptr: *mut TurboIndexRust,
    id_cstr: *const c_char,
) -> i32 {
    if index_ptr.is_null() {
        return 0;
    }
    let id_str = match from_c_str(id_cstr) {
        Some(s) => s,
        None => return 0,
    };
    let index = unsafe { &mut *index_ptr };
    if index.remove(id_str) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn little_core_turbo_index_size(index_ptr: *mut TurboIndexRust) -> usize {
    if index_ptr.is_null() {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.vectors.len()
}

#[no_mangle]
pub extern "C" fn little_core_turbo_index_search(
    index_ptr: *mut TurboIndexRust,
    query_ptr: *const f32,
    query_len: usize,
    top_k: usize,
    threshold: f32,
    coarse_multiplier: usize,
) -> *mut c_char {
    if index_ptr.is_null() || query_ptr.is_null() {
        let empty: Vec<turbo::SearchResultRust> = vec![];
        return to_c_string(&empty);
    }
    let index = unsafe { &*index_ptr };
    if query_len != index.dimension {
        let empty: Vec<turbo::SearchResultRust> = vec![];
        return to_c_string(&empty);
    }
    let query = unsafe { std::slice::from_raw_parts(query_ptr, query_len) };
    if query.iter().any(|value| !value.is_finite())
        || !threshold.is_finite()
        || !(-1.0..=1.0).contains(&threshold)
        || coarse_multiplier == 0
    {
        return to_c_string(&Vec::<turbo::SearchResultRust>::new());
    }
    let results = index.search(query, top_k, threshold, coarse_multiplier);
    to_c_string(&results)
}

#[no_mangle]
pub extern "C" fn little_core_turbo_rotate(
    index_ptr: *mut TurboIndexRust,
    vector_ptr: *const f32,
    vector_len: usize,
    out_ptr: *mut f32,
) -> i32 {
    if index_ptr.is_null() || vector_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    if vector_len != index.dimension {
        return 0;
    }
    let vector = unsafe { std::slice::from_raw_parts(vector_ptr, vector_len) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, vector_len) };
    if vector.iter().any(|value| !value.is_finite()) {
        return 0;
    }
    index.rotate(vector, out);
    1
}

// ---------------------------------------------------------------------------
// Persistent Zero-Copy Graph Index Handle (Eliminates Per-Query JSON Dumps)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct GraphIndexRust {
    forward_by_pred: HashMap<String, HashMap<String, Vec<String>>>,
    backward_by_pred: HashMap<String, HashMap<String, Vec<String>>>,
    edge_count: usize,
}

impl GraphIndexRust {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.forward_by_pred.clear();
        self.backward_by_pred.clear();
        self.edge_count = 0;
    }

    pub fn add_edge(&mut self, predicate: &str, subject: &str, object: &str) {
        let fwd = self
            .forward_by_pred
            .entry(predicate.to_string())
            .or_default()
            .entry(subject.to_string())
            .or_default();
        if !fwd.iter().any(|x| x == object) {
            fwd.push(object.to_string());
            self.edge_count += 1;
        }

        let bwd = self
            .backward_by_pred
            .entry(predicate.to_string())
            .or_default()
            .entry(object.to_string())
            .or_default();
        if !bwd.iter().any(|x| x == subject) {
            bwd.push(subject.to_string());
        }
    }

    pub fn find_ancestors_for(&self, predicate: &str, start_node: &str) -> Vec<String> {
        match self.forward_by_pred.get(predicate) {
            Some(adj) => find_ancestors(adj, start_node),
            None => Vec::new(),
        }
    }

    pub fn check_cycle_for(&self, predicate: &str, start_node: &str, target_node: &str) -> bool {
        match self.forward_by_pred.get(predicate) {
            Some(adj) => check_cycle(adj, start_node, target_node),
            None => false,
        }
    }

    pub fn bidirectional_search_for(
        &self,
        predicate: &str,
        start: &str,
        target: &str,
        max_depth: usize,
    ) -> Option<Vec<String>> {
        let fwd = self.forward_by_pred.get(predicate)?;
        let bwd = self.backward_by_pred.get(predicate)?;
        bidirectional_search(fwd, bwd, start, target, max_depth)
    }
}

#[no_mangle]
pub extern "C" fn little_core_graph_create() -> *mut GraphIndexRust {
    Box::into_raw(Box::new(GraphIndexRust::new()))
}

#[no_mangle]
pub extern "C" fn little_core_graph_destroy(graph_ptr: *mut GraphIndexRust) {
    if !graph_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(graph_ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn little_core_graph_clear(graph_ptr: *mut GraphIndexRust) {
    if !graph_ptr.is_null() {
        let graph = unsafe { &mut *graph_ptr };
        graph.clear();
    }
}

#[no_mangle]
pub extern "C" fn little_core_graph_edge_count(graph_ptr: *mut GraphIndexRust) -> usize {
    if graph_ptr.is_null() {
        return 0;
    }
    let graph = unsafe { &*graph_ptr };
    graph.edge_count
}

#[no_mangle]
pub extern "C" fn little_core_graph_add_edge(
    graph_ptr: *mut GraphIndexRust,
    predicate_cstr: *const c_char,
    subject_cstr: *const c_char,
    object_cstr: *const c_char,
) -> i32 {
    if graph_ptr.is_null() {
        return 0;
    }
    let pred = match from_c_str(predicate_cstr) {
        Some(s) if !s.is_empty() => s,
        _ => return 0,
    };
    let subj = match from_c_str(subject_cstr) {
        Some(s) if !s.is_empty() => s,
        _ => return 0,
    };
    let obj = match from_c_str(object_cstr) {
        Some(s) if !s.is_empty() => s,
        _ => return 0,
    };
    let graph = unsafe { &mut *graph_ptr };
    graph.add_edge(pred, subj, obj);
    1
}

#[no_mangle]
pub extern "C" fn little_core_graph_check_cycle(
    graph_ptr: *mut GraphIndexRust,
    predicate_cstr: *const c_char,
    start_cstr: *const c_char,
    target_cstr: *const c_char,
) -> i32 {
    if graph_ptr.is_null() {
        return 0;
    }
    let pred = match from_c_str(predicate_cstr) {
        Some(s) => s,
        None => return 0,
    };
    let start = match from_c_str(start_cstr) {
        Some(s) => s,
        None => return 0,
    };
    let target = match from_c_str(target_cstr) {
        Some(s) => s,
        None => return 0,
    };
    let graph = unsafe { &*graph_ptr };
    if graph.check_cycle_for(pred, start, target) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn little_core_graph_find_ancestors(
    graph_ptr: *mut GraphIndexRust,
    predicate_cstr: *const c_char,
    start_cstr: *const c_char,
) -> *mut c_char {
    if graph_ptr.is_null() {
        return to_c_string(&AncestorResponse { ancestors: vec![] });
    }
    let pred = match from_c_str(predicate_cstr) {
        Some(s) => s,
        None => return to_c_string(&AncestorResponse { ancestors: vec![] }),
    };
    let start = match from_c_str(start_cstr) {
        Some(s) => s,
        None => return to_c_string(&AncestorResponse { ancestors: vec![] }),
    };
    let graph = unsafe { &*graph_ptr };
    let ancestors = graph.find_ancestors_for(pred, start);
    to_c_string(&AncestorResponse { ancestors })
}

#[no_mangle]
pub extern "C" fn little_core_graph_bidirectional_search(
    graph_ptr: *mut GraphIndexRust,
    predicate_cstr: *const c_char,
    start_cstr: *const c_char,
    target_cstr: *const c_char,
    max_depth: usize,
) -> *mut c_char {
    if graph_ptr.is_null() || max_depth == 0 {
        return to_c_string(&BiSearchResponse {
            found: false,
            path: None,
        });
    }
    let pred = match from_c_str(predicate_cstr) {
        Some(s) => s,
        None => {
            return to_c_string(&BiSearchResponse {
                found: false,
                path: None,
            })
        }
    };
    let start = match from_c_str(start_cstr) {
        Some(s) => s,
        None => {
            return to_c_string(&BiSearchResponse {
                found: false,
                path: None,
            })
        }
    };
    let target = match from_c_str(target_cstr) {
        Some(s) => s,
        None => {
            return to_c_string(&BiSearchResponse {
                found: false,
                path: None,
            })
        }
    };
    let graph = unsafe { &*graph_ptr };
    match graph.bidirectional_search_for(pred, start, target, max_depth) {
        Some(p) => to_c_string(&BiSearchResponse {
            found: true,
            path: Some(p),
        }),
        None => to_c_string(&BiSearchResponse {
            found: false,
            path: None,
        }),
    }
}

// ---------------------------------------------------------------------------
// C-ABI Exports for 360° Spider-Web MLU Arena (`SpiderWebArena`)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_create(
    mitosis_min_activations: usize,
    mitosis_min_verified_spokes: usize,
) -> *mut SpiderWebArena {
    let arena = SpiderWebArena::new(
        mitosis_min_activations as u64,
        mitosis_min_verified_spokes,
    );
    Box::into_raw(Box::new(arena))
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_destroy(arena_ptr: *mut SpiderWebArena) {
    if !arena_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(arena_ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_node_count(arena_ptr: *const SpiderWebArena) -> usize {
    if arena_ptr.is_null() {
        return 0;
    }
    let arena = unsafe { &*arena_ptr };
    arena.cells.len()
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_mitosis_count(arena_ptr: *const SpiderWebArena) -> usize {
    if arena_ptr.is_null() {
        return 0;
    }
    let arena = unsafe { &*arena_ptr };
    arena.total_mitosis_events as usize
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_learn_spoke(
    arena_ptr: *mut SpiderWebArena,
    subject_ptr: *const c_char,
    predicate_ptr: *const c_char,
    object_ptr: *const c_char,
    learning_rate: f32,
    verified: i32,
) -> f32 {
    if arena_ptr.is_null()
        || subject_ptr.is_null()
        || predicate_ptr.is_null()
        || object_ptr.is_null()
    {
        return 0.0;
    }
    let (Ok(subj), Ok(pred), Ok(obj)) = (
        unsafe { CStr::from_ptr(subject_ptr) }.to_str(),
        unsafe { CStr::from_ptr(predicate_ptr) }.to_str(),
        unsafe { CStr::from_ptr(object_ptr) }.to_str(),
    ) else {
        return 0.0;
    };
    let arena = unsafe { &mut *arena_ptr };
    arena.learn_spoke_relation(subj, pred, obj, learning_rate, verified != 0)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_predict_spoke(
    arena_ptr: *const SpiderWebArena,
    concept_ptr: *const c_char,
    spoke_idx: usize,
    top_k: usize,
) -> *mut c_char {
    if arena_ptr.is_null() || concept_ptr.is_null() {
        return to_c_string(&Vec::<mlu_arena::SpokePredictionDto>::new());
    }
    let Ok(concept) = (unsafe { CStr::from_ptr(concept_ptr) }).to_str() else {
        return to_c_string(&Vec::<mlu_arena::SpokePredictionDto>::new());
    };
    let arena = unsafe { &*arena_ptr };
    let preds = arena.predict_spoke_targets(concept, spoke_idx, top_k);
    to_c_string(&preds)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_train_stream(
    arena_ptr: *mut SpiderWebArena,
    text_ptr: *const c_char,
    learning_rate: f32,
) -> *mut c_char {
    let empty = mlu_arena::StreamTrainStatsDto {
        sentences_processed: 0,
        tokens_processed: 0,
        spoke_updates: 0,
        average_surprise: 0.0,
        total_cells: 0,
        mitosis_promotions: 0,
    };
    if arena_ptr.is_null() || text_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(text) = (unsafe { CStr::from_ptr(text_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &mut *arena_ptr };
    let stats = arena.train_raw_stream(text, learning_rate);
    to_c_string(&stats)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_generate(
    arena_ptr: *const SpiderWebArena,
    prompt_ptr: *const c_char,
    max_tokens: usize,
) -> *mut c_char {
    if arena_ptr.is_null() || prompt_ptr.is_null() {
        return to_c_string(&"");
    }
    let Ok(prompt) = (unsafe { CStr::from_ptr(prompt_ptr) }).to_str() else {
        return to_c_string(&"");
    };
    let arena = unsafe { &*arena_ptr };
    let out = arena.generate_from_prompt(prompt, max_tokens);
    to_c_string(&out)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_save_snapshot(
    arena_ptr: *const SpiderWebArena,
    path_ptr: *const c_char,
) -> i32 {
    if arena_ptr.is_null() || path_ptr.is_null() {
        return 0;
    }
    let Ok(path) = (unsafe { CStr::from_ptr(path_ptr) }).to_str() else {
        return 0;
    };
    let arena = unsafe { &*arena_ptr };
    if arena.save_snapshot_file(path) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_load_snapshot(
    arena_ptr: *mut SpiderWebArena,
    path_ptr: *const c_char,
) -> i32 {
    if arena_ptr.is_null() || path_ptr.is_null() {
        return 0;
    }
    let Ok(path) = (unsafe { CStr::from_ptr(path_ptr) }).to_str() else {
        return 0;
    };
    let arena = unsafe { &mut *arena_ptr };
    if arena.load_snapshot_file(path) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_dump_json(arena_ptr: *const SpiderWebArena) -> *mut c_char {
    if arena_ptr.is_null() {
        return to_c_string(&serde_json::json!({}));
    }
    let arena = unsafe { &*arena_ptr };
    let s = arena.dump_snapshot_json();
    match CString::new(s) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_restore_json(
    arena_ptr: *mut SpiderWebArena,
    json_ptr: *const c_char,
) -> i32 {
    if arena_ptr.is_null() || json_ptr.is_null() {
        return 0;
    }
    let Ok(payload) = (unsafe { CStr::from_ptr(json_ptr) }).to_str() else {
        return 0;
    };
    let arena = unsafe { &mut *arena_ptr };
    if arena.restore_snapshot_json(payload) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_query_360_beam(
    arena_ptr: *mut SpiderWebArena,
    concept_ptr: *const c_char,
    max_hops: usize,
    top_k: usize,
) -> *mut c_char {
    let empty = mlu_arena::Query360BeamDto {
        focal_concept: String::new(),
        active_subcore: "CORE_ROOT".to_string(),
        role: "unknown".to_string(),
        activated_cells: vec!["CORE_ROOT".to_string()],
        predictions: Vec::new(),
        generated_continuation: String::new(),
        fluent_synthesis: String::new(),
    };
    if arena_ptr.is_null() || concept_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(concept) = (unsafe { CStr::from_ptr(concept_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &mut *arena_ptr };
    let res = arena.query_360_beam(concept, max_hops, top_k);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_consolidate(
    arena_ptr: *mut SpiderWebArena,
    min_transition_weight: f32,
    replay_lr: f32,
) -> *mut c_char {
    let empty = mlu_arena::ConsolidationReportDto {
        replayed_edges: 0,
        pruned_transitions: 0,
        average_replay_surprise: 0.0,
        mitosis_promotions: 0,
        total_cells: 0,
    };
    if arena_ptr.is_null() {
        return to_c_string(&empty);
    }
    let arena = unsafe { &mut *arena_ptr };
    let report = arena.consolidate_arena(min_transition_weight, replay_lr);
    to_c_string(&report)
}

#[derive(Serialize)]
struct InvariantCheckDto {
    passed: bool,
    gate: String,
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_verify_invariants(
    arena_ptr: *mut SpiderWebArena,
    subject_ptr: *const c_char,
    predicate_ptr: *const c_char,
    object_ptr: *const c_char,
) -> *mut c_char {
    let fallback = InvariantCheckDto {
        passed: false,
        gate: "INVALID_PTR".to_string(),
    };
    if arena_ptr.is_null() || subject_ptr.is_null() || predicate_ptr.is_null() || object_ptr.is_null() {
        return to_c_string(&fallback);
    }
    let Ok(subject) = (unsafe { CStr::from_ptr(subject_ptr) }).to_str() else {
        return to_c_string(&fallback);
    };
    let Ok(predicate) = (unsafe { CStr::from_ptr(predicate_ptr) }).to_str() else {
        return to_c_string(&fallback);
    };
    let Ok(object) = (unsafe { CStr::from_ptr(object_ptr) }).to_str() else {
        return to_c_string(&fallback);
    };
    let arena = unsafe { &*arena_ptr };
    let (passed, gate) = arena.verify_spoke_invariants(subject, predicate, object);
    to_c_string(&InvariantCheckDto {
        passed,
        gate: gate.to_string(),
    })
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_compare_360(
    arena_ptr: *mut SpiderWebArena,
    concept_a_ptr: *const c_char,
    concept_b_ptr: *const c_char,
    max_hops: usize,
) -> *mut c_char {
    let empty = mlu_arena::Compare360Dto {
        concept_a: String::new(),
        concept_b: String::new(),
        anchor_similarity: 0.0,
        mutually_exclusive: false,
        shared_attractors: Vec::new(),
        distinct_a: Vec::new(),
        distinct_b: Vec::new(),
        comparative_synthesis: String::new(),
    };
    if arena_ptr.is_null() || concept_a_ptr.is_null() || concept_b_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(concept_a) = (unsafe { CStr::from_ptr(concept_a_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let Ok(concept_b) = (unsafe { CStr::from_ptr(concept_b_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &mut *arena_ptr };
    let res = arena.compare_concepts_360(concept_a, concept_b, max_hops);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_discover_curiosity(
    arena_ptr: *mut SpiderWebArena,
    max_items: usize,
) -> *mut c_char {
    let empty = mlu_arena::CuriosityReportDto {
        analogical_hypotheses: Vec::new(),
        curiosity_gaps: Vec::new(),
    };
    if arena_ptr.is_null() {
        return to_c_string(&empty);
    }
    let arena = unsafe { &*arena_ptr };
    let res = arena.discover_epistemic_hypotheses(max_items);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_simulate_causal(
    arena_ptr: *mut SpiderWebArena,
    root_concept_ptr: *const c_char,
    max_hops: usize,
) -> *mut c_char {
    let empty = mlu_arena::CausalSimulationDto {
        root_concept: String::new(),
        affected_concepts: Vec::new(),
        causal_steps: Vec::new(),
        causal_synthesis: String::new(),
    };
    if arena_ptr.is_null() || root_concept_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(root_concept) = (unsafe { CStr::from_ptr(root_concept_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &mut *arena_ptr };
    let res = arena.simulate_causal_wave_360(root_concept, max_hops);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_autonomous_curiosity(
    arena_ptr: *mut SpiderWebArena,
    min_confidence: f32,
    max_promotions: usize,
) -> *mut c_char {
    let empty = mlu_arena::AutonomousCuriosityStepDto {
        hypotheses_evaluated: 0,
        hypotheses_promoted: 0,
        promoted_hypotheses: Vec::new(),
        open_questions: Vec::new(),
    };
    if arena_ptr.is_null() {
        return to_c_string(&empty);
    }
    let arena = unsafe { &mut *arena_ptr };
    let res = arena.autonomous_curiosity_step(min_confidence, max_promotions);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_intervene_counterfactual(
    arena_ptr: *mut SpiderWebArena,
    root_concept_ptr: *const c_char,
    suppressed_concept_ptr: *const c_char,
    max_hops: usize,
) -> *mut c_char {
    let empty = mlu_arena::CounterfactualInterventionDto {
        root_concept: String::new(),
        suppressed_concept: String::new(),
        baseline_affected: Vec::new(),
        invalidated_conclusions: Vec::new(),
        preserved_conclusions: Vec::new(),
        counterfactual_synthesis: String::new(),
    };
    if arena_ptr.is_null() || root_concept_ptr.is_null() || suppressed_concept_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(root_concept) = (unsafe { CStr::from_ptr(root_concept_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let Ok(suppressed_concept) = (unsafe { CStr::from_ptr(suppressed_concept_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &mut *arena_ptr };
    let res = arena.intervene_counterfactual_360(root_concept, suppressed_concept, max_hops);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_find_bridge(
    arena_ptr: *mut SpiderWebArena,
    concept_a_ptr: *const c_char,
    concept_b_ptr: *const c_char,
    max_hops: usize,
) -> *mut c_char {
    let empty = mlu_arena::RelationalBridgeDto {
        concept_a: String::new(),
        concept_b: String::new(),
        connected: false,
        hop_count: 0,
        bridge_steps: Vec::new(),
        bridge_synthesis: String::new(),
    };
    if arena_ptr.is_null() || concept_a_ptr.is_null() || concept_b_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(concept_a) = (unsafe { CStr::from_ptr(concept_a_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let Ok(concept_b) = (unsafe { CStr::from_ptr(concept_b_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &*arena_ptr };
    let res = arena.find_relational_bridge_360(concept_a, concept_b, max_hops);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_export_topology(
    arena_ptr: *mut SpiderWebArena,
    max_edges: usize,
) -> *mut c_char {
    let empty = mlu_arena::TopologyGraphDto {
        total_cells: 0,
        total_verified_edges: 0,
        nodes: Vec::new(),
        edges: Vec::new(),
        mermaid_diagram: String::new(),
    };
    if arena_ptr.is_null() {
        return to_c_string(&empty);
    }
    let arena = unsafe { &*arena_ptr };
    let res = arena.export_topology_graph(max_edges);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_discover_clusters(
    arena_ptr: *mut SpiderWebArena,
    min_members: usize,
) -> *mut c_char {
    let empty: Vec<mlu_arena::LatentClusterDto> = Vec::new();
    if arena_ptr.is_null() {
        return to_c_string(&empty);
    }
    let arena = unsafe { &*arena_ptr };
    let res = arena.discover_latent_clusters_360(min_members);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_revise_belief(
    arena_ptr: *mut SpiderWebArena,
    subject_ptr: *const c_char,
    predicate_ptr: *const c_char,
    object_ptr: *const c_char,
    new_verified: i32,
    evidence_weight: f32,
) -> *mut c_char {
    let empty = mlu_arena::BeliefRevisionDto {
        subject_concept: String::new(),
        predicate: String::new(),
        target_concept: String::new(),
        spoke: String::new(),
        old_polarity: 0,
        new_polarity: 0,
        polarity_flipped: false,
        pos_evidence: 0.0,
        neg_evidence: 0.0,
        nars_confidence: 0.0,
    };
    if arena_ptr.is_null()
        || subject_ptr.is_null()
        || predicate_ptr.is_null()
        || object_ptr.is_null()
    {
        return to_c_string(&empty);
    }
    let Ok(subject) = (unsafe { CStr::from_ptr(subject_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let Ok(predicate) = (unsafe { CStr::from_ptr(predicate_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let Ok(object) = (unsafe { CStr::from_ptr(object_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &mut *arena_ptr };
    let res = arena.revise_belief_360(
        subject,
        predicate,
        object,
        new_verified != 0,
        evidence_weight,
    );
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_deliberate_360(
    arena_ptr: *mut SpiderWebArena,
    concept_ptr: *const c_char,
    max_hops: usize,
) -> *mut c_char {
    let empty = mlu_arena::Deliberation360Dto {
        focal_concept: String::new(),
        waves_executed: 0,
        free_energy: 1.0,
        verified_facts: Vec::new(),
        causal_consequences: Vec::new(),
        cluster_peers: Vec::new(),
        analogical_insights: Vec::new(),
        follow_up_question: String::new(),
        deliberative_synthesis: String::new(),
    };
    if arena_ptr.is_null() || concept_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(concept) = (unsafe { CStr::from_ptr(concept_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &mut *arena_ptr };
    let res = arena.deliberate_360(concept, max_hops);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_plan_goal_360(
    arena_ptr: *mut SpiderWebArena,
    goal_ptr: *const c_char,
    start_ptr: *const c_char,
    max_steps: usize,
) -> *mut c_char {
    let empty = mlu_arena::GoalPlan360Dto {
        goal_concept: String::new(),
        start_concept: String::new(),
        achievable: false,
        total_steps: 0,
        plan_confidence: 0.0,
        root_prerequisites: Vec::new(),
        plan_steps: Vec::new(),
        plan_synthesis: String::new(),
    };
    if arena_ptr.is_null() || goal_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(goal) = (unsafe { CStr::from_ptr(goal_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let start = if start_ptr.is_null() {
        ""
    } else {
        (unsafe { CStr::from_ptr(start_ptr) }).to_str().unwrap_or("")
    };
    let arena = unsafe { &*arena_ptr };
    let res = arena.plan_goal_360(goal, start, max_steps);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_evaluate_order_360(
    arena_ptr: *mut SpiderWebArena,
    predicate_ptr: *const c_char,
    start_ptr: *const c_char,
    max_hops: usize,
) -> *mut c_char {
    let empty = mlu_arena::OrderWave360Dto {
        predicate: String::new(),
        start_concept: String::new(),
        poset_consistent: true,
        element_count: 0,
        ordered_chain: Vec::new(),
        transitive_pairs: Vec::new(),
        order_synthesis: String::new(),
    };
    if arena_ptr.is_null() {
        return to_c_string(&empty);
    }
    let predicate = if predicate_ptr.is_null() {
        ""
    } else {
        (unsafe { CStr::from_ptr(predicate_ptr) }).to_str().unwrap_or("")
    };
    let start = if start_ptr.is_null() {
        ""
    } else {
        (unsafe { CStr::from_ptr(start_ptr) }).to_str().unwrap_or("")
    };
    let arena = unsafe { &*arena_ptr };
    let res = arena.evaluate_order_wave_360(predicate, start, max_hops);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_summarize_domain_360(
    arena_ptr: *mut SpiderWebArena,
    domain_ptr: *const c_char,
    max_items: usize,
) -> *mut c_char {
    let empty = mlu_arena::DomainSummary360Dto {
        domain_name: String::new(),
        active_concepts_count: 0,
        verified_synapses_count: 0,
        central_hubs: Vec::new(),
        key_principles: Vec::new(),
        emergent_clusters: Vec::new(),
        open_curiosity_questions: Vec::new(),
        domain_synthesis: String::new(),
    };
    if arena_ptr.is_null() {
        return to_c_string(&empty);
    }
    let domain = if domain_ptr.is_null() {
        ""
    } else {
        (unsafe { CStr::from_ptr(domain_ptr) }).to_str().unwrap_or("")
    };
    let arena = unsafe { &*arena_ptr };
    let res = arena.summarize_domain_360(domain, max_items);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_diagnose_abductive_360(
    arena_ptr: *mut SpiderWebArena,
    observations_ptr: *const c_char,
    max_hops: usize,
    top_k: usize,
) -> *mut c_char {
    let empty = mlu_arena::AbductiveDiagnosis360Dto {
        observations: Vec::new(),
        best_explanation: String::new(),
        best_coverage_ratio: 0.0,
        candidates: Vec::new(),
        diagnostic_synthesis: String::new(),
    };
    if arena_ptr.is_null() || observations_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(observations) = (unsafe { CStr::from_ptr(observations_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &*arena_ptr };
    let res = arena.diagnose_abductive_360(observations, max_hops, top_k);
    to_c_string(&res)
}

#[no_mangle]
pub extern "C" fn little_core_mlu_arena_compose_discourse_360(
    arena_ptr: *mut SpiderWebArena,
    concept_ptr: *const c_char,
    max_hops: usize,
) -> *mut c_char {
    let empty = mlu_arena::Discourse360Dto {
        focal_concept: String::new(),
        title: String::new(),
        paragraph_count: 0,
        paragraphs: Vec::new(),
        full_discourse: String::new(),
    };
    if arena_ptr.is_null() || concept_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(concept) = (unsafe { CStr::from_ptr(concept_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let arena = unsafe { &mut *arena_ptr };
    let res = arena.compose_discourse_360(concept, max_hops);
    to_c_string(&res)
}

// ---------------------------------------------------------------------------
// True End-to-End Differentiable Neural 360° Spider-Web MLU FFI
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn little_core_neural_mlu_create(seed: u64) -> *mut NeuralSpiderWebMlu {
    Box::into_raw(Box::new(NeuralSpiderWebMlu::new(seed)))
}

/// # Safety
/// `ptr` must be a valid pointer returned by `little_core_neural_mlu_create`.
#[no_mangle]
pub unsafe extern "C" fn little_core_neural_mlu_free(ptr: *mut NeuralSpiderWebMlu) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// # Safety
/// `ptr` must be valid, `text_ptr` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn little_core_neural_mlu_train(
    ptr: *mut NeuralSpiderWebMlu,
    text_ptr: *const c_char,
    epochs: usize,
    lr: f32,
) -> *mut c_char {
    let empty = neural_mlu::NeuralTrainReportDto {
        epochs: 0,
        sequences_trained: 0,
        total_tokens: 0,
        vocab_size: 0,
        initial_loss: 0.0,
        final_loss: 0.0,
        initial_perplexity: 1.0,
        final_perplexity: 1.0,
        optimizer_steps: 0,
    };
    if ptr.is_null() || text_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(text) = (unsafe { CStr::from_ptr(text_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let model = unsafe { &mut *ptr };
    let rep = model.train_text_corpus(text, epochs, lr);
    to_c_string(&rep)
}

/// # Safety
/// `ptr` must be valid, `prompt_ptr` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn little_core_neural_mlu_generate(
    ptr: *mut NeuralSpiderWebMlu,
    prompt_ptr: *const c_char,
    max_new_tokens: usize,
) -> *mut c_char {
    let empty = neural_mlu::NeuralGenerationDto {
        prompt: String::new(),
        generated_text: String::new(),
        neural_thinking: String::new(),
        neural_answer: String::new(),
        generated_tokens: Vec::new(),
        mean_token_confidence: 0.0,
        known_prompt_tokens: 0,
        total_prompt_tokens: 0,
        active_spoke_weights: Vec::new(),
        dominant_logic_gates: Vec::new(),
    };
    if ptr.is_null() || prompt_ptr.is_null() {
        return to_c_string(&empty);
    }
    let Ok(prompt) = (unsafe { CStr::from_ptr(prompt_ptr) }).to_str() else {
        return to_c_string(&empty);
    };
    let model = unsafe { &*ptr };
    let res = model.generate_response(prompt, max_new_tokens);
    to_c_string(&res)
}

/// # Safety
/// `ptr` must be a valid pointer returned by `little_core_neural_mlu_create`.
#[no_mangle]
pub unsafe extern "C" fn little_core_neural_mlu_dump_json(
    ptr: *mut NeuralSpiderWebMlu,
) -> *mut c_char {
    if ptr.is_null() {
        return CString::new("{}").unwrap().into_raw();
    }
    let model = unsafe { &*ptr };
    let s = serde_json::to_string(model).unwrap_or_else(|_| "{}".to_string());
    CString::new(s)
        .unwrap_or_else(|_| CString::new("{}").unwrap())
        .into_raw()
}

/// # Safety
/// `ptr` must be valid, `json_ptr` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn little_core_neural_mlu_restore_json(
    ptr: *mut NeuralSpiderWebMlu,
    json_ptr: *const c_char,
) -> bool {
    if ptr.is_null() || json_ptr.is_null() {
        return false;
    }
    let Ok(json_str) = (unsafe { CStr::from_ptr(json_ptr) }).to_str() else {
        return false;
    };
    let Ok(restored) = serde_json::from_str::<NeuralSpiderWebMlu>(json_str) else {
        return false;
    };
    let model = unsafe { &mut *ptr };
    *model = restored;
    true
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_ancestors() {
        let mut adj = HashMap::new();
        adj.insert("sparrow".to_string(), vec!["bird".to_string()]);
        adj.insert("bird".to_string(), vec!["animal".to_string()]);
        adj.insert("animal".to_string(), vec!["living_thing".to_string()]);

        let ancestors = find_ancestors(&adj, "sparrow");
        assert_eq!(ancestors.len(), 3);
        assert!(ancestors.contains(&"bird".to_string()));
        assert!(ancestors.contains(&"animal".to_string()));
        assert!(ancestors.contains(&"living_thing".to_string()));
    }

    #[test]
    fn test_cycle_detection() {
        let mut adj = HashMap::new();
        adj.insert("A".to_string(), vec!["B".to_string()]);
        adj.insert("B".to_string(), vec!["C".to_string()]);
        adj.insert("C".to_string(), vec!["A".to_string()]);

        assert!(check_cycle(&adj, "A", "C"));
        assert!(check_cycle(&adj, "C", "A"));
        assert!(!check_cycle(&adj, "A", "D"));
    }

    #[test]
    fn test_topological_sort() {
        let nodes = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let edges = vec![
            ("A".to_string(), "B".to_string()),
            ("B".to_string(), "C".to_string()),
        ];
        let order = topological_sort(&nodes, &edges).unwrap();
        assert_eq!(order, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_turbo_index_search() {
        let signs = vec![1.0; 512];
        let mut index = TurboIndexRust::new(512, &signs).expect("valid signs");
        let mut v1 = vec![0.0f32; 512];
        v1[0] = 1.0;
        index.add("v1", &v1, "{}");

        let mut v2 = vec![0.0f32; 512];
        v2[1] = 1.0;
        index.add("v2", &v2, "{}");

        let results = index.search(&v1, 5, 0.5, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
        assert!(results[0].score > 0.9);
    }

    #[test]
    fn turbo_rejects_invalid_dimension_without_panicking() {
        let signs = [1.0f32; 512];
        for dimension in [0, 32, 513, 1024] {
            assert!(little_core_turbo_index_create(dimension, signs.as_ptr(), dimension).is_null());
        }
        let bad_signs = [0.0f32; 512];
        assert!(little_core_turbo_index_create(512, bad_signs.as_ptr(), 512).is_null());
        assert!(TurboIndexRust::new(32, &[1.0; 32]).is_err());
    }

    #[test]
    fn test_persistent_graph_index_rust() {
        let mut g = GraphIndexRust::new();
        g.add_edge("is_a", "WHALE", "MAMMAL");
        g.add_edge("is_a", "MAMMAL", "ANIMAL");
        g.add_edge("is_a", "ANIMAL", "LIVING_THING");
        assert_eq!(g.edge_count, 3);
        let anc = g.find_ancestors_for("is_a", "WHALE");
        assert_eq!(anc.len(), 3);
        assert!(g.check_cycle_for("is_a", "WHALE", "LIVING_THING"));
        assert!(!g.check_cycle_for("is_a", "LIVING_THING", "WHALE"));
        let path = g
            .bidirectional_search_for("is_a", "WHALE", "LIVING_THING", 6)
            .expect("path");
        assert_eq!(path, vec!["WHALE", "MAMMAL", "ANIMAL", "LIVING_THING"]);
    }

    #[test]
    fn test_spiderweb_arena_rust_liquid_and_mitosis() {
        let mut arena = SpiderWebArena::new(3, 2);
        let s1 = arena.learn_spoke_relation("ORCA", "is_a", "CETACEAN", 0.25, true);
        let s2 = arena.learn_spoke_relation("ORCA", "has_part", "DORSAL_FIN", 0.25, true);
        let s3 = arena.learn_spoke_relation("ORCA", "can", "ECHOLOCATE", 0.25, true);
        let _s4 = arena.learn_spoke_relation("CETACEAN", "is_a", "MAMMAL", 0.25, true);
        let _s5 = arena.learn_spoke_relation("MAMMAL", "is_a", "ANIMAL", 0.25, true);
        assert!(s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0);
        assert_eq!(arena.total_mitosis_events, 1);
        let orca_id = arena.name_to_id["ORCA"] as usize;
        assert_eq!(arena.cells[orca_id].role, "subcore");

        let preds = arena.predict_spoke_targets("ORCA", 0, 5);
        assert!(!preds.is_empty());
        assert!(preds
            .iter()
            .any(|p| p.target_concept == "CETACEAN" && p.combined_confidence > 0.5));

        let stream_stats1 = arena.train_raw_stream(
            "An orca is a marine cetacean that hunts in coordinated pods. \
             Orcas use echolocation clicks to navigate deep ocean waters.",
            0.25,
        );
        let stream_stats2 = arena.train_raw_stream(
            "An orca is a marine cetacean that hunts in coordinated pods. \
             Orcas use echolocation clicks to navigate deep ocean waters.",
            0.25,
        );
        assert!(stream_stats1.tokens_processed >= 16);
        assert!(stream_stats2.average_surprise < stream_stats1.average_surprise);

        let gen = arena.generate_from_prompt("an orca", 6);
        assert!(gen.contains("marine") || gen.contains("cetacean"));

        // Phase 1: Native Rust 360° Multi-Hop Beam Traversal
        let beam = arena.query_360_beam("ORCA", 3, 10);
        assert_eq!(beam.focal_concept, "ORCA");
        assert_eq!(beam.role, "subcore");
        assert!(beam.predictions.iter().any(|p| p.target_concept == "ANIMAL"));
        assert!(beam.activated_cells.contains(&"CETACEAN".to_string()));
        assert!(beam.activated_cells.contains(&"MAMMAL".to_string()));

        // Phase 2: Native Rust Sleep-Cycle Consolidation + JSON Snapshot round-trip
        let cons = arena.consolidate_arena(0.05, 0.15);
        assert!(cons.replayed_edges >= 5);
        let snap_json = arena.dump_snapshot_json();
        let mut arena2 = SpiderWebArena::new(3, 2);
        assert!(arena2.restore_snapshot_json(&snap_json));
        assert_eq!(arena2.cells.len(), arena.cells.len());
    }

    #[test]
    fn test_true_differentiable_neural_mlu_backprop_and_generation() {
        let mut model = NeuralSpiderWebMlu::new(42);
        let corpus = "is a tiger a mammal? yes a tiger is a warm blooded mammal. \
                      is a tiger a vehicle? no a tiger is a living animal not a vehicle. \
                      what causes crop failure? drought causes crop failure and famine. \
                      how to achieve wisdom? curiosity and study lead to knowledge and wisdom.";
        let rep = model.train_text_corpus(corpus, 35, 0.035);
        assert!(rep.final_loss < rep.initial_loss * 0.25);
        assert!(rep.final_loss < 0.35);

        let gen1 = model.generate_response("is a tiger a mammal?", 10);
        assert!(gen1.generated_text.to_lowercase().contains("yes"));
        assert!(gen1.generated_text.to_lowercase().contains("mammal"));

        let gen2 = model.generate_response("is a tiger a vehicle?", 12);
        assert!(gen2.generated_text.to_lowercase().contains("no"));

        let gen3 = model.generate_response("what causes crop failure?", 10);
        assert!(gen3.generated_text.to_lowercase().contains("drought"));
    }
}



