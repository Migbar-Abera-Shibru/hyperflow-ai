"""

"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

class ParameterSchema(BaseModel):
    """  
    Extracted parameter schema from OpenAPI.
    
    Handles both simple parameters (query, path) and
    complex request bodies (JSON Schema).
    """
    name: str
    type: str
    description: str = ""
    required: bool = False
    schema: Dict[str, Any] = Field(default_facotry=dict)
    location: str = "body" # a path, a query, a header or a body

    @property
    def type_hint(self) -> str:
        """ convert JSON schema type to Python type hint."""

        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict"
        }

        # check for nested types
        if self.type == "array" and "items" in self.schema:
            item_type = self.schema["items"].get("type", "Any")
            return f"List[{type_map.get(item_type, 'Any')}]"

        return type_map.get(self.type, "Any")

class ToolDefinition(BaseModel):
    """
    Extracted tool definition from OpenAPI.
    
    Represents a complete API operation with:
    - HTTP method and path
    - Input parameters
    - Output schemas
    - Operation description
    """

    name: str
    path: str
    method: str
    operation_id: Optional[str] = None
    summary: str = ""
    input_parameters: List[ParameterSchema] = Field(default_factory=list)
    output_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        if self.operation_id:
            return self.operation_id
        return f"{self.method}_{self.path.replace('/','_').strip('_')}"

    @property
    def required_inputs(self) -> List[ParameterSchema]:
        """Get Required input parameters """
        return [p for p in self.input_parameters if p.required]

    @property
    def optional_inputs(self) -> List[ParameterSchema]:
        """get the optional input parameters"""
        return [p for p in self.input_parameters if not p.required]

