"""
tree.py
-------
RoutingTree, TreeNode, and LeafNode classes.
Implements recursive metadata-based traversal with default path fallback.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LeafNode:
    """A leaf node containing FAQ and guardrail vector stores."""
    faq_store: Any
    guardrail_store: Any
    name: str = "unnamed_leaf"


@dataclass
class TreeNode:
    """An internal or leaf node in the routing tree."""
    metadata: dict[str, Any]
    children: list[TreeNode] = field(default_factory=list)
    default_child: Optional[TreeNode] = None
    leaf: Optional[LeafNode] = None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf."""
        return self.leaf is not None

    def route(self, query_meta: dict[str, Any]) -> TreeNode:
        """Route a query to the appropriate child node based on metadata matching."""
        best_match: Optional[TreeNode] = None
        best_score: int = 0

        for child in self.children:
            match_score = 0
            is_match = True
            for key, value in child.metadata.items():
                if key in query_meta:
                    if query_meta[key] == value:
                        match_score += 1
                    else:
                        is_match = False
                        break
            if is_match and match_score > best_score:
                best_score = match_score
                best_match = child

        if best_match is not None:
            return best_match
        elif self.default_child is not None:
            return self.default_child
        else:
            return self


class RoutingTree:
    """A tree-structured routing system for query navigation."""

    def __init__(self, root: TreeNode):
        self.root = root

    def traverse(self, query_meta: dict[str, Any]) -> LeafNode:
        """Recursively traverse the tree to find the appropriate LeafNode."""
        return self._traverse_recursive(self.root, query_meta)

    def _traverse_recursive(self, node: TreeNode, query_meta: dict[str, Any]) -> LeafNode:
        """Internal recursive traversal."""
        if node.is_leaf():
            return node.leaf

        next_node = node.route(query_meta)

        if next_node is node:
            if node.leaf is not None:
                return node.leaf
            raise RuntimeError(
                f"Traversal stuck at node with metadata {node.metadata}."
            )

        return self._traverse_recursive(next_node, query_meta)