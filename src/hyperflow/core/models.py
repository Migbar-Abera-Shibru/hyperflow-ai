"""
core data models for hyperflow ai

This module defines the foundational data structures for:
- Tool-Schema hypergraph representation
- schema nodes( input, output, effects, conditions)
- hyperedges ( tools connecting inputs to outputs)
- Dependencies( port-level schema links)
- support matrices for fast lookup


These models are designed to be: 
- type-safe with Pydantic validation
- JSON serializable for API responses
- Compatible with NetworkX for graph operations.
- efficient for sparse matrix operations

Production Considerations: 
- All models use immutable data structure where possible
- Validation prevents malformed hypergraphs
- Support matrices are precomputed for performance 
"""

from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
import json
from pydantic import BaseModel, Field, ConfigDict, validator
import numpy as np
from dataclasses import dataclass

class NodeType(str, Enum):
    """ Types of nodes in the hypergraph"""
    INPUT_SCHEMA = "input_schema"
    OUTPUT_SCHEMA = "output_schema"
    EFFECT = "effect"
    CONDITION = "condition"

class Node(BaseModel):

    """
    A node in the tool-schema hypergraph
    
    Nodes:
     input schemas: parameters that tools require as input
     output schemas: values that tools produce
     effects: changes of state in the environment
     conditions: preconditions for tool execution
     
    Attributes:
        id :  unique node identifier
        name: human readable name
        node_type: types of node
        description: detailed description of the schema
        json_schema: JSON schema specification
        type_hint: python type hint( for type checking)
        is_required
        default_value: optional default value
        metadata: metadata for extensibility
        """
    model_config = ConfigDict(
        frozed=True,
        json_schema_extra={"examples": [{
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "user_id",
            "node_type": "input_schema",
            "description": "Unique identifier for a user",
            "json_schema": {"type": "integer"},
            "type_hint": "int",
            "is_required": True
        }]}

    )

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    node_type: NodeType
    description: str = Field(..., min_length=1)
    json_schema: Dict[str, Any] = Field(default_factory=dict)
    type_hint: Optional[str] = None
    is_required: bool = True
    default_value: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('json_schema')
    def validate_schema(cls, v:Dict[str, Any]) -> Dict[str, Any]:
        """ ensure the JSON schema is valid"""
        if v and '$schema' not in v:
            v['$schema']= 'http://json-schema.org/draft-07/schema#'
        return v

    def to_network_node(self) -> Dict[str, Any]:
        """convert to networkX node attributes"""
        return {
            'name': self.name,
            'type': self.node_type.value,
            'description' : self.description,
            'is_required': self.is_required
        }

    def __hash__(self) -> int:
        """hash by id for use in sets/dicts"""
        return hash(self.id)
class HyperEdge(BaseModel):
    """
    A hyperedge representing tool in the hypergraph.
    Unlike regular edges that connects two nodes, hyperedges connect 
    multiple input nodes to multiple output nodes. This captures
    the joint constraints of tool invocation: all required inputs 
    must be availble, and all outputs are produced together.

    Attributes: 
        id: Unique hyperedge identifier
        name: Tool name
        description: Tool description
        input_nodes: Set of input schema node IDs
        output_nodes: Set of output/effect node IDs
        metadata: Additional tool metadata (OpenAPI spec, etc.)
    """
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    input_nodes: Set[UUID] = Field(default_factory=set)
    output_nodes: Set[UUID] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('input_nodes')
    def validate_inputs(cls, v: Set[UUID]) -> Set[UUID]:
        """Tools must have atleast one imput"""
        if not v:
            raise ValueError("Tool must have at least one input node")
        return v

    @validator('output_nodes')
    def validate_outputs(cls, v: Set[UUID]) -> Set[UUID]:
        """Tools must have at least one output."""
        if not v:
            raise ValueError("Tool must have at least one output node")
        return v

    def get_inputs(self) -> List[UUID]:
        """ get input node IDs as list for deterministic ordering"""
        return sorted(list(self.input_nodes))

    def get_outputs(self) -> List[UUID]:
        """Get output node IDs as list for deterministic ordering."""
        return sorted(list(self.output_nodes))

    def to_network_edge(self) -> Dict[str, Any]:
        """Convert to networkx hyperedge representation"""
        return {
            'name': self.name,
            'description': self.description,
            'is_hyperedge': True
        }

class Dependency(BaseModel):
    """
    A port-level schema dependency between nodes.
    
    Represents that an output schema of one tool can satisfy an
    input schema of another tool. Dependencies are directional
    and have weights indicating semantic match strength.
    
    Attributes:
        id: Unique dependency identifier
        source_node: Output/effect node that provides data
        target_node: Input/condition node that requires data
        weight: Semantic match strength (0.0 to 1.0)
        is_automated: Whether this was automatically inferred
        verified: Whether this has been human-verified
    """
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    source_node = UUID
    target_node = UUID
    weight = float = Field(..., ge=0, le=1.0)
    is_automated = bool = True
    verified = bool = False
    created_at = datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_weak(self) -> bool:
        """check if this is a weak dependency"""
        return self.weight < 0.5


class ToolSchemaHypergraph(BaseModel):
    """
    The complete tool-schema hypergraph

    This is the core data structure that models all tools and their
    schema level relationships. It enables efficient retrival of task relevant tools,
    deficit oriented expansion for support graphs and schema aware planning and execution.

    Attributes:
        nodes: All schema, effect, and condition nodes
        hyperedges: All tool hyperedges
        dependencies: All port-level schema dependencies
        support_matrix: Precomputed tool-schema support scores
        node_index: Fast lookup for nodes by ID and name
        edge_index: Fast lookup for hyperedges by ID and name
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: Dict[UUID, Node] = Field(default_factory=dict)
    hyperedges: Dict[UUID, HyperEdge] = Field(default_factory=dict)
    dependencies : Dict[UUID, Dependency] = Field(default_factory=dict)

    # private fields ( not serialized to JSON)
    _support_matrix: Optional[np.ndarray] = None
    _node_name_index: Dict[str, UUID] = Field(default_factory=dict, exclude=True)
    _edge_name_index: Dict[str, UUID] = Field(default_factory=dict, exclude=True)

    def __init__(self, **data):
        # initialize with automatic indexing
        super().__init__(**data)
        self.build_indexes()

    def _build_indexs(self) -> None:
        # build a fast lookup indexes for node and edge

        self._node_name_index = {
            node.name: node_id 
            for node_id, node in self.nodes.items()
        }
        self._edge_name_index = {
            edge.name: edge_id 
            for edge_id, edge in self.hyperedges.items()
        }

    def add_node(self, node: Node) -> None:
        """ Add a node to the hypergraph"""
        self.nodes[node.id] = node
        self._node_name_index[node.name] = node.id

    def add_hyperedge(self, edge: HyperEdge) -> None:
        # validate if all referenced nodes exist
        for node_id in edge.input_nodes | edge.output_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found in hypergraph")

        self.hyperedges[edge.id]= edge
        self._edge_name_index[edge.name] = edge.id

    def add_dependenc(self, dep: Dependency) -> None:
        # validate if node exists
        if dep.source_node not in self.nodes:
            raise ValueError(f"Source node {dep.source_node} not found")
        if dep.target_node not in self.nodes:
            raise ValueError(f"Target node {dep.target_node} not found")

        self.dependencies[dep.id] = dep

    def get_node_by_name(self, name: str) -> Optional[Node]:
        # get the node by its name
        node_id = self._node_name_index.get(name)
        return self.nodes.get(node_id) if node_id else None

    def get_hyperedge_by_name(slef, name: str) -> Optional[HyperEdge]:
        # get the hyperedge by its name
        edge_id = self._edge_name_index.get(name)
        return self.nodes.get(edge_id) if edge_id else None

    def get_producers_for_input(
            self,
            input_node_id: UUID,
            min_weight: float = 0.0

    ) -> List[Tuple[UUID, float]]:
        # get all producer tools that can satisfy an input schema, returns list of (hyperedge_id, weight) typles
        producers = []
        for dep in self.dependencies.values():
            if dep.target_node == input_node_id and dep.weight > min_weight:
                # find wiich hyperedge produces the source
                for edge_id, edge in self.hyperedges.items():
                    if dep.source_node in edge.output_nodes:
                        producers.append((edge_id, min_weight))
                        break

        return producers

    def get_consumers_for_output(
            self,
            output_node_id: UUID,
            min_weight: float =  0.0
    ) -> List[Tuple[UUID, float]]:
        # get all consumer tools that require an output schema

        consumers = []
        for dep in self.dependencies.values():
            if dep.source_node == output_node_id and dep.weight >= min_weight:
                # find which tool consumes the target
                for edge_id, edge in self.hyperedges.items():
                    if dep.target_node in edge.input_nodes:
                        consumers.append((edge_id, dep.weight))
                        break
        return consumers

    def build_support_matrix(self) -> np.ndarray:
        """
        Build the tool schema support matrix.

        Matrix Dimentions; are ( num_input_nodes, num_hyperedges)
        Each entry: max support weight from edge outputs to input node
        """
        input_nodes = [
            n for n in self.nodes.values()
            if n.node_type == NodeType.INPUT_SCHEMA
        ]

        edge_list = list(self.hyperedges.values())

        support_matrix = np.zeros((len(input_nodes), len(edge_list)))

        for i, input_node in enumerate(input_nodes):
            producers = self.get_producers_for_input(input_node.id)
            for edge_id, weight in producers:
                edge_index = next(
                    idx for idx, e in enumerate(edge_list)
                    if e.id == edge_id
                )
                support_matrix[i, edge_index] = max(support_matrix[i, edge_index], weight)

        return support_matrix
            




