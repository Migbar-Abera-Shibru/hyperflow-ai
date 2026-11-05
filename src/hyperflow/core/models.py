"""
Core data models for HyperFlow AI.

This module defines the foundational data structures for:
- Tool-Schema Hypergraph representation
- Schema nodes (inputs, outputs, effects, conditions)
- HyperEdges (tools connecting inputs to outputs)
- Dependencies (port-level schema links)

Fully Pydantic v2 compatible.
"""

from typing import Dict, List, Optional, Set, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
import json
import numpy as np

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    PrivateAttr,
    field_validator,
)


class NodeType(str, __import__("enum").Enum):
    """Types of nodes in the hypergraph."""
    INPUT_SCHEMA = "input_schema"
    OUTPUT_SCHEMA = "output_schema"
    EFFECT = "effect"
    CONDITION = "condition"


class Node(BaseModel):
    """
    A node in the Tool-Schema Hypergraph.
    
    Nodes represent input schemas, output schemas, effects, or conditions.
    """
    model_config = ConfigDict(frozen=True)
    
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
    
    @field_validator('json_schema')
    @classmethod
    def validate_schema(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure JSON Schema is valid."""
        if v and '$schema' not in v:
            v = dict(v)
            v['$schema'] = 'http://json-schema.org/draft-07/schema#'
        return v
    
    def to_networkx_node(self) -> Dict[str, Any]:
        """Convert to NetworkX node attributes."""
        return {
            'name': self.name,
            'type': self.node_type.value,
            'description': self.description,
            'is_required': self.is_required
        }
    
    def __hash__(self) -> int:
        return hash(self.id)


class HyperEdge(BaseModel):
    """
    A hyperedge representing a tool in the hypergraph.
    
    Connects multiple input nodes to multiple output nodes.
    """
    model_config = ConfigDict(frozen=True)
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    input_nodes: Set[UUID] = Field(default_factory=set)
    output_nodes: Set[UUID] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('input_nodes')
    @classmethod
    def validate_inputs(cls, v: Set[UUID]) -> Set[UUID]:
        if not v:
            raise ValueError("Tool must have at least one input node")
        return v
    
    @field_validator('output_nodes')
    @classmethod
    def validate_outputs(cls, v: Set[UUID]) -> Set[UUID]:
        if not v:
            raise ValueError("Tool must have at least one output node")
        return v
    
    def get_inputs(self) -> List[UUID]:
        return sorted(list(self.input_nodes))
    
    def get_outputs(self) -> List[UUID]:
        return sorted(list(self.output_nodes))
    
    def to_networkx_edge(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'is_hyperedge': True
        }


class Dependency(BaseModel):
    """
    A port-level schema dependency between nodes.
    """
    model_config = ConfigDict(frozen=True)
    
    id: UUID = Field(default_factory=uuid4)
    source_node: UUID
    target_node: UUID
    weight: float = Field(..., ge=0.0, le=1.0)
    is_automated: bool = True
    verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_weak(self) -> bool:
        return self.weight < 0.5


class ToolSchemaHypergraph(BaseModel):
    """
    The complete Tool-Schema Hypergraph.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    nodes: Dict[UUID, Node] = Field(default_factory=dict)
    hyperedges: Dict[UUID, HyperEdge] = Field(default_factory=dict)
    dependencies: Dict[UUID, Dependency] = Field(default_factory=dict)
    
    # Private attributes (Pydantic v2 style)
    _support_matrix: Optional[np.ndarray] = PrivateAttr(default=None)
    _node_name_index: Dict[str, UUID] = PrivateAttr(default_factory=dict)
    _edge_name_index: Dict[str, UUID] = PrivateAttr(default_factory=dict)
    
    def model_post_init(self, __context: Any) -> None:
        """Rebuild indexes after Pydantic initialization."""
        self._build_indexes()
    
    def _build_indexes(self) -> None:
        """Build fast lookup indexes."""
        self._node_name_index = {
            node.name: node_id
            for node_id, node in self.nodes.items()
        }
        self._edge_name_index = {
            edge.name: edge_id
            for edge_id, edge in self.hyperedges.items()
        }
    
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self._node_name_index[node.name] = node.id
    
    def add_hyperedge(self, edge: HyperEdge) -> None:
        for node_id in edge.input_nodes | edge.output_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found in hypergraph")
        self.hyperedges[edge.id] = edge
        self._edge_name_index[edge.name] = edge.id
    
    def add_dependency(self, dep: Dependency) -> None:
        if dep.source_node not in self.nodes:
            raise ValueError(f"Source node {dep.source_node} not found")
        if dep.target_node not in self.nodes:
            raise ValueError(f"Target node {dep.target_node} not found")
        self.dependencies[dep.id] = dep
    
    def get_node_by_name(self, name: str) -> Optional[Node]:
        node_id = self._node_name_index.get(name)
        return self.nodes.get(node_id) if node_id else None
    
    def get_hyperedge_by_name(self, name: str) -> Optional[HyperEdge]:
        edge_id = self._edge_name_index.get(name)
        return self.hyperedges.get(edge_id) if edge_id else None
    
    def get_producers_for_input(
        self,
        input_node_id: UUID,
        min_weight: float = 0.0
    ) -> List[tuple]:
        producers = []
        for dep in self.dependencies.values():
            if dep.target_node == input_node_id and dep.weight >= min_weight:
                for edge_id, edge in self.hyperedges.items():
                    if dep.source_node in edge.output_nodes:
                        producers.append((edge_id, dep.weight))
                        break
        return producers
    
    def get_consumers_for_output(
        self,
        output_node_id: UUID,
        min_weight: float = 0.0
    ) -> List[tuple]:
        consumers = []
        for dep in self.dependencies.values():
            if dep.source_node == output_node_id and dep.weight >= min_weight:
                for edge_id, edge in self.hyperedges.items():
                    if dep.target_node in edge.input_nodes:
                        consumers.append((edge_id, dep.weight))
                        break
        return consumers
    
    def build_support_matrix(self) -> np.ndarray:
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
                support_matrix[i, edge_index] = max(
                    support_matrix[i, edge_index],
                    weight
                )
        
        return support_matrix
    
    def get_subgraph(self, edge_ids: Set[UUID]) -> 'ToolSchemaHypergraph':
        node_ids: Set[UUID] = set()
        for edge_id in edge_ids:
            edge = self.hyperedges[edge_id]
            node_ids.update(edge.input_nodes)
            node_ids.update(edge.output_nodes)
        
        deps = {
            dep_id: dep for dep_id, dep in self.dependencies.items()
            if dep.source_node in node_ids and dep.target_node in node_ids
        }
        
        return ToolSchemaHypergraph(
            nodes={nid: self.nodes[nid] for nid in node_ids},
            hyperedges={eid: self.hyperedges[eid] for eid in edge_ids},
            dependencies=deps
        )
    
    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ToolSchemaHypergraph':
        data = json.loads(json_str)
        return cls(**data)
    
    def to_networkx(self):
        import networkx as nx
        G = nx.MultiDiGraph()
        for node_id, node in self.nodes.items():
            G.add_node(str(node_id), **node.to_networkx_node())
        for edge_id, edge in self.hyperedges.items():
            edge_attrs = edge.to_networkx_edge()
            for input_id in edge.input_nodes:
                for output_id in edge.output_nodes:
                    G.add_edge(
                        str(input_id),
                        str(output_id),
                        **edge_attrs,
                        edge_type='hyperedge'
                    )
        for dep in self.dependencies.values():
            G.add_edge(
                str(dep.source_node),
                str(dep.target_node),
                weight=dep.weight,
                edge_type='dependency',
                verified=dep.verified
            )
        return G
    
    def validate(self) -> List[str]:
        issues = []
        for edge in self.hyperedges.values():
            for node_id in edge.input_nodes | edge.output_nodes:
                if node_id not in self.nodes:
                    issues.append(
                        f"Hyperedge {edge.name} references missing node {node_id}"
                    )
        for dep in self.dependencies.values():
            if dep.source_node not in self.nodes:
                issues.append(
                    f"Dependency {dep.id} references missing source node"
                )
            if dep.target_node not in self.nodes:
                issues.append(
                    f"Dependency {dep.id} references missing target node"
                )
        return issues


class SupportMatrix:
    """Precomputed support matrix for fast lookup."""
    
    def __init__(
        self,
        matrix: np.ndarray,
        input_node_ids: List[UUID],
        hyperedge_ids: List[UUID]
    ):
        self.matrix = matrix
        self.input_node_ids = input_node_ids
        self.hyperedge_ids = hyperedge_ids
        self.input_index = {nid: i for i, nid in enumerate(input_node_ids)}
        self.edge_index = {eid: j for j, eid in enumerate(hyperedge_ids)}
    
    def get_producers(
        self,
        input_node_id: UUID,
        min_weight: float = 0.0
    ) -> List[tuple]:
        if input_node_id not in self.input_index:
            return []
        row = self.input_index[input_node_id]
        scores = self.matrix[row]
        producers = []
        for col, score in enumerate(scores):
            if score >= min_weight:
                producers.append((self.hyperedge_ids[col], float(score)))
        return sorted(producers, key=lambda x: x[1], reverse=True)
    
    def get_top_producers(
        self,
        input_node_id: UUID,
        k: int = 5,
        min_weight: float = 0.0
    ) -> List[tuple]:
        return self.get_producers(input_node_id, min_weight)[:k]
    
    def get_support_score(
        self,
        hyperedge_id: UUID,
        input_node_id: UUID
    ) -> float:
        if hyperedge_id not in self.edge_index:
            return 0.0
        if input_node_id not in self.input_index:
            return 0.0
        row = self.input_index[input_node_id]
        col = self.edge_index[hyperedge_id]
        return float(self.matrix[row, col])