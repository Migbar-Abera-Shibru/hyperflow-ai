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
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from hyperflow.builders.open_api_parser import SchemaExtractor, ToolDefinition
from hyperflow.core.models import HyperEdge, Node, NodeType, ToolSchemaHypergraph


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

    def _extract_nodes(
            self,
            tool_defs: List[ToolDefinition],
    ) -> Dict[str, Node]:
        """
        Extract all nodes from tool definitions.
        
        Returns a mapping from node name to Node object.
        """

        nodes = {}

        for tool in tool_defs:
            # input nodes
            for node in SchemaExtractor.extract_input_nodes(tool):
                nodes[node.name] = node

            # output nodes
            for node in SchemaExtractor.extract_output_nodes(tool):
                nodes[node.name] = node

        return nodes

    def _create_hyperedge(
            self, tool_defs: List[ToolDefinition],
            nodes_by_name: Dict[str, Node]
    ) -> Dict[str, HyperEdge]:
        """
        Create hyperedges from tool definitions.
        """
        edges = {}

        for tool in tool_defs:
            # get input node ids
            input_node_ids = set()
            for param in tool.input_parameters:
                node_name = f"{tool.name}_{param.name}"
                if node_name in nodes_by_name:
                    input_node_ids.add(nodes_by_name[node_name].id)

            # get the output node ids
            output_node_ids = set()
            #from explicit output schemas
            for status_code, schema in tool.output_schemas.items():
                if schema.get('type') == 'object':
                    for prop_name in schema.get('properties', {}):
                        node_name = f"{tool.name}_{prop_name}"
                        if node_name in nodes_by_name:
                            output_node_ids.add(nodes_by_name[node_name].id)

                else:
                    node_name = f"{tool.name}_output"
                    if node_name in nodes_by_name:
                        output_node_ids.add(nodes_by_name[node_name].id)

            # ensure we have atleast one output
            if not output_node_ids and tool.output_schemas:
                # create a generic output node
                output_node = Node(
                    name=f"{tool_name}_output",
                    node_type=NodeType.OUTPUT_SCHEMA,
                    description=f"Output from {tool.name}",
                    json_schema=tool.output_schemas.get('200', {}),
                    metadata={"tool_name":tool.name}
                )

                nodes_by_name[output_node.name] = output_node
                output_node_ids.add(output_node.id)

            # create the hyperedge
            edge = HyperEdge(
                name=tool.full_name,
                description=tool.desciption or tool.summary,
                input_nodes=input_node_ids,
                output_nodes=output_node_ids,
                metadata={
                    'path': tool.path,
                    'method': tool.method,
                    'tags': tool.tags,
                    'operation_id': tool.operation_id
                }
            )

            edges[tool.full_name] = edge
        return edges

    def _infer_dependencies(
            self,
            tool_defs: List[ToolDefinition],
            nodes_by_name: Dict[str, Node],
            edges_by_name: Dict[str, HyperEdge],
    ) -> List[Tuple[UUID, UUID, float]]:
        """
        Infer dependencies between nodes.
        
        Uses:
        1. Type matching
        2. Name similarity
        3. Semantic similarity (if available)
        """
        dependencies = []

        # for each input, find matching outputs
        for input_name, input_node in nodes_by_name.items():
            if input_node.node_type != NodeType.INPUT_SCHEMA:
                continue

            # skip if this an input to a tool that has no input
            input_edge_name = input_node.metadata.get('tool_name')
            if not input_edge_name:
                continue

            for output_name, output_node in nodes_by_name.items():
                if output_node.node_type != NodeType.OUTPUT_SCHEMA:
                    continue

                # skip same tool
                if input_edge_name == output_node.metadata.get('tool_name'):
                    continue

                # check type compatability
                if not self._types_compatable(input_node, output_node):
                    continue

                # calculate similarity
                similarity = self._calculate_similarity(input_node, output_node)

                if similarity >= 0.6:
                    weight = min(1.0, similarity)
                    dependencies.append(output_node.id, input_node.id, weight)

        return dependencies




        



