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
from hyperflow.core.models import Dependency, HyperEdge, Node, NodeType, ToolSchemaHypergraph


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

    def _types_compatible(self, input_node: Node, output_node: Node) -> bool:
        """ check if input and output types are compatible. """
        # if we have type hints, use them
        if input_node.type_hint and output_node.type_hint:
            # convert type hints to sets for matching
            input_type = input_node.type_hint.lower()
            output_type = output_node.type_hint.lower()

            # Direct match 
            if input_type == output_type:
                return True

            # special cases
            if input_type == 'str' and output_type in ['str', 'string']:
                return True

            if input_type == 'int' and output_type in ['int', 'integer']:
                return True

            if input_type == 'float' and output_type in ['float', 'number']:
                return True

            if input_type == 'dict' and output_type in ['dict', 'object']:
                return True

            if input_type == 'list' and output_type in ['list', 'array']:
                return True

        # fallback to JSON schema types
        input_schema = input_node.json_schema
        output_schema = output_node.json_schema

        if input_schema and output_schema:
            if input_schema.get('type') == output_schema.get('type'):
                return True

        return False

    def _calculate_similarity(self, input_node: Node, output_node: Node) -> float:
        """ calculate similarity between an input and output node """
        # name overlap
        input_words = set(input_node.name.lower().split('_'))
        output_words = set(output_node.name.lower().split('_'))

        common = input_words & output_words
        union = input_words | output_words

        if union:
            name_similarity = len(common) / len(union)
        else: 
            name_similarity = 0.0

        # description overlap ( if descriptions exist)
        if input_node.description and output_node.description:
            input_desc_words = set(input_node.description.lower().split())
            output_desc_words = set(output_node.description.lower().split())

            common_desc = input_desc_words & output_desc_words
            union_desc = input_desc_words | output_desc_words

            if union_desc:
                desc_similarity = len(common_desc) / len(union_desc)
            else: 
                desc_similarity = 0.0

        else:
            desc_similarity = 0.0

        # combine similarities 
        similarity = 0.6 * name_similarity + 0.4 * desc_similarity

        return similarity

    def _add_semantic_dependencies(
            self,
            nodes_by_name: Dict[str, Node],
            edges_by_name: Dict[str, HyperEdge]
    ) -> List[Tuple[UUID, UUID, float]]:
        """
        Add semantic dependencies using embeddings.
        """
        if not self.use_embeddings or not self.embedding_model:
            return []

        dependencies = []

        # get all input and output nodes
        input_nodes = [n for n in nodes_by_name.values()
                        if n.node_type == NodeType.INPUT_SCHEMA]
        output_nodes = [n for n in nodes_by_name.values()
                        if n.node_type == NodeType.OUTPUT_SCHEMA]

        if not input_nodes or not output_nodes:
            return []

        # generate descriptions for embedding 
        input_texts = [f"{n.name}: {n.description}" for n in input_nodes]
        output_texts = [f"{n.name}: {n.description}" for n in output_nodes]

        try:
            # compute embeddings
            input_embeddings = self.embedding_model.encode(input_texts)
            output_embeddings = self.embedding_model.encode(output_texts)

            # compute similarities
            import numpy as np
            similarities = np.dot(input_embeddings, output_embeddings)

            # for each input, find top output matches
            for i, input_node in enumerate(input_nodes):
                for j, output_node in enumerate(output_nodes):
                    sim = similarities[i, j]

                    # skip if low similarity
                    if sim < 0.5:
                        continue

                    # skip same tool
                    if input_node.metadata.get('tool_name') == output_node.metadata.get('tool_name'):
                        continue

                    # add as dependency with semantic weight
                    weight = max(0.3, sim * 0.8) # scale down a bit
                    dependencies.append((output_node.id, input_node.id, weight))

        except Exception as e:
            logger.warning(f"Failed to compute semantic dependencies: {e}")

        return dependencies

    def _filter_dependencies(
            self,
            dependencies: List[Tuple[UUID, UUID, float]],
            hypergraph: ToolSchemaHypergraph
    ) -> List[Dependency]:
        """
        Filter and deduplicate dependencies.
        """
        # group by (source, target) and keep max weight
        dep_map= {}

        for source_id, target_id, weight in dependencies:
            key = (source_id, target_id)
            if key not in dep_map or weight > dep_map[key]:
                dep_map[key] = weight

        # create dependency objects
        filtered_deps = []

        for (source_id, target_id), weight in dep_map.items():
            # check if nodes exist
            if source_id not in hypergraph.nodes:
                logger.debug(f"Skipping dependency: source {source_id} not found")
                continue

            if target_id not in hypergraph.nodes:
                logger.debug(f"Skipping dependency: target {target_id} not found")
                continue

            # check if this is a valid dependency
            source_node = hypergraph.nodes[source_id]
            target_node = hypergraph.nodes[target_id]

            # only allow if target is input/condition and source is output/effect
            if target_node.node_type not in [NodeType.INPUT_SCHEMA, NodeType.CONDITION]:
                continue

            if source_node.node_type not in [NodeType.OUTPUT_SCHEMA, NodeType.EFFECT]:
                continue

            dep = Dependency(
                source_node=source_id,
                target_node=target_id,
                weight=weight,
                is_automated=True,
                verified=False
            )

            filtered_deps.append(dep)

        return filtered_deps
        
    
    def _build_support_matrix(
            self,
            hypergraph: ToolSchemaHypergraph
    ) -> None:
        """
        Build and attach the support matrix.
        """
        from hyperflow.core.models import SupportMatrix

        input_nodes = [n for n in hypergraph.nodes.values()
                       if n.node_type == NodeType.INPUT_SCHEMA]

    




        



