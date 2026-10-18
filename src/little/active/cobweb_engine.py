"""Fisher's Cobweb Incremental Conceptual Clustering Engine.

Organizes concepts into a probabilistic hierarchy by evaluating 4 operators:
1. Best Host Insertion: place instance into the best existing child cluster.
2. New Class Creation: create a new singleton cluster at the current level.
3. Merge: combine the two best host clusters into a single composite hypernym (Hypernym Induction).
4. Split: dissolve the best host cluster and promote its children (Boundary Cleavage).

All decisions are guided by Cobweb Category Utility (CU):
  CU({C_1, ..., C_K}) = (1/K) * sum_{k=1}^K P(C_k) * sum_i sum_j [ P(A_i = V_{ij} | C_k)^2 - P(A_i = V_{ij})^2 ]
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from little.core.models import generate_id


@dataclass
class CobwebInstance:
    """An observation instance passed into the Cobweb conceptual clustering tree."""

    concept_id: str
    features: dict[str, set[str]] = field(default_factory=dict)
    continuous_vector: list[float] | None = None

    def get_feature_values(self, attribute: str) -> set[str]:
        return self.features.get(attribute, set())

    @property
    def attributes(self) -> set[str]:
        return set(self.features.keys())


class CobwebNode:
    """A probabilistic concept node in the Cobweb conceptual hierarchy."""

    def __init__(self, node_id: str | None = None, parent: CobwebNode | None = None) -> None:
        self.node_id: str = node_id or generate_id("cnode")
        self.parent: CobwebNode | None = parent
        self.children: list[CobwebNode] = []
        self.concept_count: int = 0
        self.concept_ids: set[str] = set()

        # attribute -> value -> count of instances having this value
        self.attribute_counts: dict[str, dict[str, int]] = {}

        # Optional latent concept label if this node has been reified
        self.latent_label: str | None = None
        self.category_utility: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_instance(self, instance: CobwebInstance) -> None:
        """Incorporate an instance's feature counts into this node."""
        if instance.concept_id in self.concept_ids:
            raise ValueError(f"Duplicate concept ID {instance.concept_id!r}")
        self.concept_count += 1
        self.concept_ids.add(instance.concept_id)

        for attr, values in instance.features.items():
            if attr not in self.attribute_counts:
                self.attribute_counts[attr] = {}
            for val in values:
                self.attribute_counts[attr][val] = self.attribute_counts[attr].get(val, 0) + 1

    def remove_instance(self, instance: CobwebInstance) -> None:
        """Remove an instance's feature counts from this node."""
        if instance.concept_id in self.concept_ids:
            self.concept_ids.remove(instance.concept_id)
            self.concept_count = max(0, self.concept_count - 1)

            for attr, values in instance.features.items():
                if attr in self.attribute_counts:
                    for val in values:
                        if val in self.attribute_counts[attr]:
                            self.attribute_counts[attr][val] -= 1
                            if self.attribute_counts[attr][val] <= 0:
                                del self.attribute_counts[attr][val]
                    if not self.attribute_counts[attr]:
                        del self.attribute_counts[attr]

    def probability_attribute_value(self, attr: str, val: str) -> float:
        """P(A_i = V_ij | C_k): Conditional probability of value given this cluster."""
        if self.concept_count == 0:
            return 0.0
        val_count = self.attribute_counts.get(attr, {}).get(val, 0)
        return val_count / self.concept_count

    def expected_correct_guesses(self) -> float:
        """sum_i sum_j P(A_i = V_ij | C_k)^2."""
        if self.concept_count == 0:
            return 0.0
        total = 0.0
        for attr, vals in self.attribute_counts.items():
            for val, count in vals.items():
                prob = count / self.concept_count
                total += prob * prob
        return total

    def get_shared_features(self, min_ratio: float = 0.6) -> dict[str, list[str]]:
        """Return features that are dominant in this concept node."""
        shared: dict[str, list[str]] = {}
        if self.concept_count == 0:
            return shared

        for attr, vals in self.attribute_counts.items():
            for val, count in vals.items():
                ratio = count / self.concept_count
                if ratio >= min_ratio:
                    shared.setdefault(attr, []).append(val)
        return shared


class CobwebTree:
    """Incremental Cobweb conceptual clustering tree with 4 operators."""

    def __init__(self, category_utility_threshold: float = 0.25) -> None:
        if not math.isfinite(category_utility_threshold) or category_utility_threshold < 0:
            raise ValueError("Category utility threshold must be finite and nonnegative")
        self.root: CobwebNode = CobwebNode(node_id="cnode_root")
        self.cu_threshold: float = category_utility_threshold
        self.all_instances: dict[str, CobwebInstance] = {}

    def calculate_category_utility(
        self,
        parent: CobwebNode,
        children: list[CobwebNode],
    ) -> float:
        """Computes Cobweb Category Utility across a set of child clusters."""
        if not children or parent.concept_count == 0:
            return 0.0

        num_children = len(children)
        parent_expected = parent.expected_correct_guesses()

        child_term = 0.0
        for child in children:
            if child.concept_count == 0:
                continue
            pk = child.concept_count / parent.concept_count
            child_expected = child.expected_correct_guesses()
            child_term += pk * (child_expected - parent_expected)

        cu = child_term / num_children
        return max(0.0, cu)

    def insert(self, instance: CobwebInstance) -> CobwebNode:
        """Incrementally insert an instance into the Cobweb tree."""
        if instance.concept_id in self.all_instances:
            raise ValueError(f"Duplicate concept ID {instance.concept_id!r}")
        stored = copy.deepcopy(instance)
        self.all_instances[stored.concept_id] = stored
        result = self._cobweb(self.root, stored)
        def refresh(node):
            for child in node.children:
                refresh(child)
            node.category_utility = self.calculate_category_utility(node, node.children)
        refresh(self.root)
        return result

    def _cobweb(self, current: CobwebNode, instance: CobwebInstance) -> CobwebNode:
        """Recursive Cobweb insertion evaluating 4 operators."""
        # 1. Base case: leaf node
        if current.is_leaf:
            if current.concept_count == 0:
                # Empty root/leaf: simply incorporate
                current.add_instance(instance)
                return current

            # Node already has 1 instance: create 2 children (one for existing, one for new)
            existing_instance_id = next(iter(current.concept_ids))
            existing_inst = self.all_instances.get(existing_instance_id)

            # Create child 1 for existing instance
            child1 = CobwebNode(parent=current)
            if existing_inst:
                child1.add_instance(existing_inst)
            else:
                # Reconstruct child1 from current counts
                for attr, vals in current.attribute_counts.items():
                    for val, count in vals.items():
                        child1.attribute_counts.setdefault(attr, {})[val] = count
                child1.concept_count = current.concept_count
                child1.concept_ids = set(current.concept_ids)

            # Create child 2 for new instance
            child2 = CobwebNode(parent=current)
            child2.add_instance(instance)

            current.children = [child1, child2]
            current.add_instance(instance)
            current.category_utility = self.calculate_category_utility(current, current.children)
            return child2

        # 2. Non-leaf node: evaluate all 4 operators
        current.add_instance(instance)

        # Candidate 1: Best child insertion
        best_child, best_score, second_best_child, second_best_score = self._evaluate_children(
            current, instance
        )

        # Candidate 2: Create new singleton child at this level
        score_new_child = self._evaluate_create_child(current, instance)

        # Candidate 3: Merge top-2 best children
        score_merge = -1.0
        if len(current.children) >= 2 and best_child and second_best_child:
            score_merge = self._evaluate_merge(current, best_child, second_best_child, instance)

        # Candidate 4: Split best child
        score_split = -1.0
        if best_child and not best_child.is_leaf:
            score_split = self._evaluate_split(current, best_child)

        # Determine winner among candidates
        best_operator = "insert"
        max_score = best_score

        if score_new_child > max_score:
            max_score = score_new_child
            best_operator = "create"

        if score_merge > max_score:
            max_score = score_merge
            best_operator = "merge"

        if score_split > max_score:
            max_score = score_split
            best_operator = "split"

        # Execute chosen operator
        if best_operator == "insert" and best_child:
            return self._cobweb(best_child, instance)

        elif best_operator == "create":
            new_child = CobwebNode(parent=current)
            new_child.add_instance(instance)
            current.children.append(new_child)
            current.category_utility = self.calculate_category_utility(current, current.children)
            return new_child

        elif best_operator == "merge" and best_child and second_best_child:
            merged_node = self._execute_merge(current, best_child, second_best_child)
            return self._cobweb(merged_node, instance)

        elif best_operator == "split" and best_child:
            self._execute_split(current, best_child)
            # Remove instance from current before re-evaluating split level
            current.remove_instance(instance)
            return self._cobweb(current, instance)

        # Default fallback
        if best_child:
            return self._cobweb(best_child, instance)
        new_child = CobwebNode(parent=current)
        new_child.add_instance(instance)
        current.children.append(new_child)
        return new_child

    def _evaluate_children(
        self,
        current: CobwebNode,
        instance: CobwebInstance,
    ) -> tuple[CobwebNode | None, float, CobwebNode | None, float]:
        """Test inserting instance into each child and return top-2 best children and CU."""
        if not current.children:
            return None, -1.0, None, -1.0

        scores: list[tuple[float, CobwebNode]] = []

        for child in current.children:
            # Trial statistics are detached; never add/remove an existing ID
            # in the live tree during evaluation.
            trial = copy.copy(child)
            trial.concept_ids = set(child.concept_ids)
            trial.attribute_counts = copy.deepcopy(child.attribute_counts)
            if instance.concept_id not in trial.concept_ids:
                trial.add_instance(instance)
            trial_children = [trial if candidate is child else candidate for candidate in current.children]
            cu = self.calculate_category_utility(current, trial_children)
            scores.append((cu, child))

        scores.sort(key=lambda item: item[0], reverse=True)

        best_child = scores[0][1]
        best_score = scores[0][0]
        second_child = scores[1][1] if len(scores) > 1 else None
        second_score = scores[1][0] if len(scores) > 1 else -1.0

        return best_child, best_score, second_child, second_score

    def _evaluate_create_child(
        self,
        current: CobwebNode,
        instance: CobwebInstance,
    ) -> float:
        """Calculate CU if a new singleton child is created."""
        temp_child = CobwebNode(parent=current)
        temp_child.add_instance(instance)

        trial_children = current.children + [temp_child]
        cu = self.calculate_category_utility(current, trial_children)
        return cu

    def _evaluate_merge(
        self,
        current: CobwebNode,
        child1: CobwebNode,
        child2: CobwebNode,
        instance: CobwebInstance,
    ) -> float:
        """Calculate CU if top-2 children are merged into a composite super-node."""
        merged = CobwebNode(parent=current)
        for c in (child1, child2):
            merged.concept_count += c.concept_count
            merged.concept_ids.update(c.concept_ids)
            for attr, vals in c.attribute_counts.items():
                for val, cnt in vals.items():
                    merged.attribute_counts.setdefault(attr, {})[val] = (
                        merged.attribute_counts.setdefault(attr, {}).get(val, 0) + cnt
                    )
        merged.add_instance(instance)

        remaining_children = [c for c in current.children if c not in (child1, child2)]
        trial_children = remaining_children + [merged]
        cu = self.calculate_category_utility(current, trial_children)
        return cu

    def _execute_merge(
        self,
        current: CobwebNode,
        child1: CobwebNode,
        child2: CobwebNode,
    ) -> CobwebNode:
        """Merge child1 and child2 into a new intermediate hypernym node."""
        merged = CobwebNode(parent=current)
        for c in (child1, child2):
            merged.concept_count += c.concept_count
            merged.concept_ids.update(c.concept_ids)
            for attr, vals in c.attribute_counts.items():
                for val, cnt in vals.items():
                    merged.attribute_counts.setdefault(attr, {})[val] = (
                        merged.attribute_counts.setdefault(attr, {}).get(val, 0) + cnt
                    )
        merged.children = [child1, child2]
        child1.parent = merged
        child2.parent = merged

        current.children = [c for c in current.children if c not in (child1, child2)]
        current.children.append(merged)
        current.category_utility = self.calculate_category_utility(current, current.children)
        return merged

    def _evaluate_split(
        self,
        current: CobwebNode,
        best_child: CobwebNode,
    ) -> float:
        """Calculate CU if best_child is dissolved and its children are promoted."""
        if not best_child.children:
            return -1.0

        remaining = [c for c in current.children if c != best_child]
        trial_children = remaining + best_child.children
        cu = self.calculate_category_utility(current, trial_children)
        return cu

    def _execute_split(
        self,
        current: CobwebNode,
        best_child: CobwebNode,
    ) -> None:
        """Dissolve best_child and promote its children to current level."""
        current.children = [c for c in current.children if c != best_child]
        for grand_child in best_child.children:
            grand_child.parent = current
            current.children.append(grand_child)
        current.category_utility = self.calculate_category_utility(current, current.children)

    def get_all_clusters(self, min_size: int = 2) -> list[CobwebNode]:
        """Collect clusters meeting size and parent-partition utility thresholds.

        A homogeneous cluster can have zero internal CU; its parent partition
        determines whether that cluster is meaningfully separated from siblings.
        """
        clusters: list[CobwebNode] = []

        def traverse(node: CobwebNode) -> None:
            if not node.is_leaf and node != self.root:
                if len(node.concept_ids) >= min_size and node.parent is not None:
                    cluster_gain = max(
                        0.0,
                        node.expected_correct_guesses() - node.parent.expected_correct_guesses(),
                    )
                    effective_cu = max(
                        node.parent.category_utility,
                        cluster_gain,
                    )
                    if effective_cu >= self.cu_threshold:
                        clusters.append(node)
            for child in node.children:
                traverse(child)

        traverse(self.root)
        return clusters
