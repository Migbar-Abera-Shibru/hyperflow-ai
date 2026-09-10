"""
Hypergraph Builder for HyperFlow AI.

This module constructs the complete Tool-Schema Hypergraph
from tool definitions (OpenAPI parsed or manually specified).
Key capabilities:
- Build nodes from tool schemas
- Create hyperedges for each tool
- Infer and add dependencies
- Validate graph integrity
- Support incremental updates

Production considerations:
- Handles duplicate tool names gracefully
- Ensures consistent node naming
- Logs construction issues
- Supports caching of built graphs
- Version control for graph evolution

Usage:
    builder = HypergraphBuilder()
    hypergraph = builder.build(tool_defs)
    builder.save(hypergraph, "hypergraph.json")
"""

import logging
from typing import List, Optional, Tuple

from hyperflow.builders.open_api_parser import ToolDefinition
from hyperflow.core.models import ToolSchemaHypergraph


logger = logging.getLogger(__name__)

class HypergraphBuilder:
    """
    Builds a Tool-Schema Hypergraph from tool definitions.
    
    The builder follows these steps:
    1. Extract nodes from all tool definitions
    2. Create hyperedges for each tool
    3. Infer dependencies between tools
    4. Add semantic dependencies (via embeddings)
    5. Validate the complete graph
    
    Supports both automated and manual construction.
    """

    def __init__(self, use_embeddings: bool = True):
        """
        Initialize the builder.
        
        Args:
            use_embeddings: Whether to use embeddings for semantic dependencies
        """
        self.use_embeddings = use_embeddings
        self.use_embeddings = None

        if use_embeddings:
            self.__init__embedding_model()

    def _init_embedding_model(self):
        """ inititalize the embedding model for semantic matching"""
        try:
            from sentence_transformers import SentenceTransformer
            self._init_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
        except ImportError:
            logger.warning(
                "sentence-transformer not installed."
                "falling back to simple matching"
            )
            self.use_embeddings = False

    def build(
            self,
            tool_defs: List[ToolDefinition],
            manual_dependencies: Optional[List[Tuple[str, str, float]]] = None
    ) -> ToolSchemaHypergraph:
        """
        Build a hypergraph from tool definitions.
        
        Args:
            tool_defs: List of tool definitions
            manual_dependencies: Manual dependencies (source, target, weight)
        
        Returns:
            Complete ToolSchemaHypergraph
        """
        logger.info(f"Building hypergraphh from {len(tool_defs)} tools")

        hypergraph = ToolSchemaHypergraph()

        # step 1 is to extract and add nodes
        nodes_by_tool = self._extract_nodes(tool_defs)

        for node in nodes_by_tool.values():
            hypergraph.add_node(node)

        logger.info(f"Added {len(hypergraph.nodes)} nodes")

        # step 2 is to create the hyperedges
        edges_by_tool = self._create_hyperedges(tool_defs, nodes_by_tool)
        for edge in edges_by_tool.values():
            hypergraph.add_hyperedge(edge)

        logger.info(f"Added {len(hypergraph.hyperedges)} hyperedges")

        # step 3 is to infer dependencies
        inferred_deps = self._infer_dependencies(tool_defs, nodes_by_tool, edges_by_tool)

        # step 4 is to add manual dependencies
        if manual_dependencies:
            inferred_deps.extend(manual_dependencies)

        # step 5 is adding semantic dependencies
        if self.use_embeddings:
            semantic_deps = self._add_semantic_dependencies(
                nodes_by_tool,
                edges_by_tool
            )
            inferred_deps.extend(semantic_deps)

        # step 6 is filter and add dependencies
        filtered_deps = self._filter_dependencies(inferred_deps, hypergraph)
        for dep in filtered_deps:
            hypergraph.add_dependency(dep)

        logger.info(f"Added {len(hypergraph.dependencies)} dependencies")

        # step 7 is to validate the graph
        issues = hypergraph.validate()
        if issues:
            logger.warning(f" Validation issue found: {issues}")

        # step 8 is to build the support matrix
        self._build_support_matrix(hypergraph)

        return hypergraph
        