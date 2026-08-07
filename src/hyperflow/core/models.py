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

