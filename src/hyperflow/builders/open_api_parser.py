"""

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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

        def _extract_base_url(self, spec: Dict[str, Any]) -> None:
            """ Extract the base url for the API"""
            servers = spec.get('servers', [])
            if servers:
                server_url = servers[0].get('url', '')
                parsed = urlparse(server_url)
                self.base_url = f"{parsed.scheme}://{parsed.netloc}"

        def _validate_spec(self, spec: Dict[str, Any]) -> None:
            """ validation for OpenAPI specification"""
            # check for required fields
            required_fields = ['openapi', 'info', 'paths']
            for field in required_fields:
                if field not in spec:
                    raise ValueError(f"Missing requried field: {field}")


            #check openAI version
            version = spec.get('openapi', '')
            if not version.startswith('3.'):
                raise ValueError(f"Unsupported OpenAPI version: {version}")

        def _parse_operation(
                self,
                path: str,
                method: str,
                operation: Dict[str, Any]
        ) -> Optional[ToolDefinition]:
            """
            Parse a single operation from the OpenAPI spec into a ToolDefinition.
            """
            try:
                # extract basic info
                tool = ToolDefinition(
                    name = self._generate_tool_name(path, method, operation),
                    path=path,
                    method=method.upper(),
                    operation_id=operation.get('operationId'),
                    summary=operation.get('summary',''),
                    description=operation.get('description',''),
                    tags=operation.get('tags', [])
                )

                #parse parameters
                parameters = operation.get('parameters', [])
                for param in parameters:
                    if '$ref' in param:
                        param = self._resolve_reference(param['$ref'])
                    schema = self._parse_parameter(param, path, method)
                    if schema:
                        tool.input_parameters.append(schema)

                # parse request body
                request_body = operation.get('requestBody')
                if request_body:
                    if "$ref" in request_body:
                        request_body= self._resolve_reference(request_body['$ref'])
                    body_schemas = self._parse_request_body(request_body)
                    tool.input_parameters.extend(body_schemas)

                # parse responses
                responses = operation.get('responses', [])
                for status_code, response in responses.items():
                    if '$ref' in response:
                        response = self._resolve_reference(response['$ref'])
                    if status_code.startswith('2'):
                        tool.output_schemas[status_code] = self._parse_response(response)

                return tool


            except Exception as e: 
                logger.error(f" Error parsing operation {method} {path}: {e}")
                return None

        def _generate_tool_name(
                self,
                path: str,
                method: str,
                operation: Dict[str, Any]
        ) -> str:
            """ Generate a unique, readable tool name"""
            # Prefer operation ID
            if 'operationId' in operation:
                return operation['operationId']

            # build from path and method 
            path_parts = [p for p in path.split('/') if p and not p.startswith('{')]
            path_parts.append(method.upper())

            # Camelcase the parts
            tool_name = ''.join(p.capitalize()for p in path_parts)
            return tool_name

        def _parse_parameter(
                self,
                param: Dict[str, Any],
                path: str,
                method: str
        ) -> Optional[ParameterSchema]:

            """
            Parse an OpenAPI parameter.
            """

            param_in = param.get('in', 'body')
            param_name = param.get('name', '')

            # extract schema 
            schema = param.get('schema', {})
            if '$ref' in schema:
                schema = self._resolve_reference(schema['$ref'])

            # determine the type
            param_type = schema.get('type', 'string')
            if param_type == 'array':
                # try to get item type
                items = schema.get('items', {})
                if '$ref' in items:
                    items = self._resolve_reference(items['$ref'])
                param_type = f"array[{items.get('type', 'any')}]"

            return ParameterSchema(
                name=param_name,
                type=param_type,
                description=param.get('description', ''),
                required=param.get('required', False),
                schema=schema,
                location=param_in
            )

        def _parse_request_body(
                self,
                request_body: Dict[str, Any]
        ) -> List[ParameterSchema] :
            """ Parse a requt body into parameter schemas"""

            schemas = []

            content = request_body.get('content', {})
            for media_type, media_spec in content.items():
                schema = media_spec.get('schema', {})

                # handle refs
                if '$ref' in schema:
                    schema = self._resolve_reference(schema['$ref'])

                # for object types, extract each property as a paramter
                if schema.get('type') == 'object':
                    properties = schema.get('properties', {})
                    required = schema.get('required', [])

                    for prop_name, prop_schema in properties.items():
                        if '$ref' in prop_schema:
                            prop_schema = self._resolve_reference(prop_schema['$ref'])


                        schemas.append(ParameterSchema(
                            name=prop_name,
                            type=prop_schema.get('type', 'string'),
                            description=prop_name in required,
                            schema=prop_schema,
                            location='body'
                        ))

                else:
                    # simple type
                    schemas.append(ParameterSchema(
                        name='body',
                        type=schema.get('type', 'object'),
                        description=request_body.get('description', ''),
                        required=request_body.get('required', ''),
                        schema=schema,
                        location='body'
                    ))

            return schemas

        def _parse_response(
                self,
                response: Dict[str, Any]
        ) -> Dict[str, Any]:
            """ parse a response into a schema"""

            schema = {}

            content = response.get('content', {})
            for media_type, media_spec in content.items():
                schema = media_spec.get('schema', {})

                if '$ref' in schema:
                    schema = self._resolve_reference(schema['$ref'])
                break # use the first media type

            return schema

        def _resolve_reference(self, ref: str) -> Dict[str, Any]:
            """ resolve a json reference 
            support both local ($ref: "#/components/schemas/User") and remote references.
            """
            if ref.startswith("#"):
                # local reference
                path_parts = ref[2:].split('/')
                current = self.spec
                for part in path_parts:
                    current = current.get(part, {})
                return current


            # for remote reference
            # cache resolved refs for simplicity
            if ref not in self.resolved_refs:
                try:
                    response = requests.get(ref)
                    response.raise_for_status()
                    self.resolved_refs[ref] = response.json()
                except Exception as e:
                    logger.warning(f"Failed to resolve remote reference {ref}: {e}")
                    return {}

            return self.resolved_refs.get(ref, {})

        def parse_with_annotations(
                self,
                spec: Dict[str, Any],
                annotations: Dict[str, Dict[str, Any]]
        ) -> List[ToolDefinition]:

            """
            Parse an OpenAPI spec with manual annotations.
        
            Annotations can override or enhance the automatic parsing.
            
            Args:
                spec: OpenAPI specification
                annotations: {
                    "tool_name": {
                        "description": "Custom description",
                        "inputs": {"param_name": {"description": "..."}},
                        "outputs": {"response": {"schema": {...}}}
                    }
                }
            
            """

            tools = self.parse_dict(spec)

            for tool in tools:
                if tool.name in annotations:
                    tool_anno = annotations[tool.name]

                    # override description
                    if 'description' in tool_anno:
                        tool.description = tool_anno['description']

                    # override input descriptions
                    if 'inputs' in tool_anno:
                        for param in tool.input_parameters:
                            if param.name in tool_anno['inputs']:
                                param.description = tool_anno['inputs'][param.name].get(
                                    'description', param.description
                                )

                                # override required status
                                if 'required' in tool_anno['inputs'][param.name]:
                                    param.required = tool_anno['inputs'][param.name]['required']

            return tools

class Schema_Extractor:
    


                





            
            



            


    


