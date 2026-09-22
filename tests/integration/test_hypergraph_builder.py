# tests/integration/test_hypergraph_builder.py

import pytest
from src.hyperflow.builders.openapi_parser import OpenAPIParser
from src.hyperflow.builders.hypergraph_builder import HypergraphBuilder


def test_end_to_end_build():
    """Test end-to-end hypergraph building."""
    # Sample spec
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "summary": "List users",
                    "responses": {
                        "200": {
                            "description": "User list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "users": {"type": "array"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    parser = OpenAPIParser()
    tool_defs = parser.parse_dict(spec)
    
    builder = HypergraphBuilder(use_embeddings=False)
    hypergraph = builder.build(tool_defs)
    
    assert len(hypergraph.nodes) > 0
    assert len(hypergraph.hyperedges) > 0
    assert len(hypergraph.validate()) == 0


if __name__ == "__main__":
    pytest.main([__file__])