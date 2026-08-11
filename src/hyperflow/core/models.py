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