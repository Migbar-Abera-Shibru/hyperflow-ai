"""

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import requests
import yaml


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


class OpenAPIParser:
        """
    Parser for OpenAPI specifications.
    
    Supports OpenAPI 3.0+ with:
    - Automatic reference resolution
    - Schema composition (allOf, anyOf, oneOf)
    - Parameter extraction from path, query, and body
    - Response schema extraction
    
    Usage:
        parser = OpenAPIParser()
        
        # Parse from file
        tool_defs = parser.parse_file("openapi.json")
        
        # Parse from URL
        tool_defs = parser.parse_url("https://api.example.com/openapi.json")
        
        # Parse from dict
        spec = load_openapi()
        tool_defs = parser.parse_dict(spec)
    """
        def __init__(self, validate_spec: bool = True):
            """
            Initialize the parser.
            
            Args:
                validate_spec: Whether to validate the OpenAPI spec
            """
            self.validate_spec = validate_spec
            self.spec = None
            self.base_url = None
            self.resolved_refs: Dict[str, Any] ={}

        def parse_file(self, file_path: str) -> List[ToolDefinition]:
            """
            Parse an OpenAPI specification from a file.
            
            Supports JSON and YAML formats.
            """
            path = Path(file_path)

            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix in ['.yaml', '.yml']:
                    spec = yaml.safe_load(f)
                else:
                    spec = json.load(f)

            return self.parse_dict(spec)

        def parse_url(self, url: str) -> List[ToolDefinition]:
            """
            Parse an OpenAPI specification from a URL.
            """
            response = requests.get(url)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '')
            if 'json' in content_type:
                spec = response.json()
            else:
                spec = yaml.safe_load(response.text)

            return self.parse_dict(spec)

        def parse_dict(self, spec: Dict[str, Any]) -> List[ToolDefinition]:
            """
            Parse an OpenAPI specification from a dictionary
            """
            self.spec = spec
            self._extract_base_url(spec)

            if self.validate_spec:
                self._validate_spec(spec)

            tools = []
            paths = spec.get('paths', {})

            for path, path_item in paths.items():
                for method, operation in path_item.items():
                    if method.lower() not in ['get', 'post', 'put', 'delete',
                                              'patch', 'head', 'options']:
                        continue

                    tool = self._parse_operation(path, method, operation)
                    if tool:
                        tools.append(tool)

            logger.info(f"Parsed {len(tools)} tools from OpenAPI spec")
            return tools




            


    


