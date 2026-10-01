//! High-performance Rust acceleration core for LITTLE's cognitive graph operations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// ---------------------------------------------------------------------------
// Pure Rust Graph Algorithms
// ---------------------------------------------------------------------------

/// Compute all transitive ancestors reachable from start_node via outgoing edges.
pub fn find_ancestors(
    adjacency: &HashMap<String, Vec<String>>,
    start_node: &str,
) -> Vec<String> {
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
    if signs_ptr.is_null() || signs_len != dimension {
        return std::ptr::null_mut();
    }
    let signs = unsafe { std::slice::from_raw_parts(signs_ptr, signs_len) };
    let index = Box::new(TurboIndexRust::new(dimension, signs));
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
    index.rotate(vector, out);
    1
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
        let mut index = TurboIndexRust::new(512, &signs);
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
}
