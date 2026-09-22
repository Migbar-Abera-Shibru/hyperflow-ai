#!/usr/bin/env python3
"""
Example: Building a Tool-Schema Hypergraph from OpenAPI.

This example demonstrates:
1. Parsing OpenAPI specifications
2. Building a hypergraph from tool definitions
3. Visualizing the hypergraph
4. Basic queries on the hypergraph
"""

import logging
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hyperflow.builders.openapi_parser import OpenAPIParser
from src.hyperflow.builders.hypergraph_builder import HypergraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def build_from_example_spec():
    """
    Build a hypergraph from an example OpenAPI spec.
    
    This example uses a sample spec for a payment API.
    """
    print("=" * 60)
    print("HyperFlow AI - Hypergraph Builder Example")
    print("=" * 60)
    
    # Step 1: Parse OpenAPI spec
    print("\n1. Parsing OpenAPI specification...")
    
    # Sample OpenAPI spec (in practice, you'd load from a file)
    sample_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Payment API",
            "version": "1.0.0"
        },
        "paths": {
            "/users/{user_id}": {
                "get": {
                    "operationId": "getUser",
                    "summary": "Get user details",
                    "parameters": [
                        {
                            "name": "user_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                            "description": "User ID"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "User details",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "name": {"type": "string"},
                                            "email": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/payments": {
                "post": {
                    "operationId": "sendPayment",
                    "summary": "Send a payment",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "recipient_id": {"type": "integer"},
                                        "amount": {"type": "number"},
                                        "currency": {"type": "string"}
                                    },
                                    "required": ["recipient_id", "amount"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Payment sent",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "transaction_id": {"type": "string"},
                                            "status": {"type": "string"}
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
    tool_defs = parser.parse_dict(sample_spec)
    print(f"   Parsed {len(tool_defs)} tools:")
    for tool in tool_defs:
        print(f"   - {tool.full_name} ({tool.method} {tool.path})")
    
    # Step 2: Build hypergraph
    print("\n2. Building hypergraph...")
    builder = HypergraphBuilder(use_embeddings=False)  # Disable embeddings for speed
    hypergraph = builder.build(tool_defs)
    
    print(f"   Nodes: {len(hypergraph.nodes)}")
    print(f"   Hyperedges: {len(hypergraph.hyperedges)}")
    print(f"   Dependencies: {len(hypergraph.dependencies)}")
    
    # Step 3: Display hypergraph structure
    print("\n3. Hypergraph structure:")
    print("-" * 40)
    
    print("\n   Nodes:")
    for node in hypergraph.nodes.values():
        print(f"   - {node.name} ({node.node_type.value})")
    
    print("\n   Hyperedges (Tools):")
    for edge in hypergraph.hyperedges.values():
        print(f"   - {edge.name}")
        print(f"     Inputs: {len(edge.input_nodes)}")
        print(f"     Outputs: {len(edge.output_nodes)}")
    
    print("\n   Dependencies:")
    for dep in hypergraph.dependencies.values():
        source = hypergraph.nodes[dep.source_node]
        target = hypergraph.nodes[dep.target_node]
        print(f"   - {source.name} -> {target.name} (weight: {dep.weight:.2f})")
    
    # Step 4: Query the hypergraph
    print("\n4. Query examples:")
    print("-" * 40)
    
    # Find producer tools for an input
    # Get the recipient_id input node
    recipient_input = hypergraph.get_node_by_name("sendPayment_recipient_id")
    if recipient_input:
        print(f"\n   Producers for {recipient_input.name}:")
        producers = hypergraph.get_producers_for_input(recipient_input.id)
        if producers:
            for edge_id, weight in producers:
                edge = hypergraph.hyperedges[edge_id]
                print(f"   - {edge.name} (weight: {weight:.2f})")
        else:
            print("   - No producers found (dependencies need to be inferred)")
    
    # Step 5: Export for visualization
    print("\n5. Exporting to NetworkX for visualization...")
    nx_graph = hypergraph.to_networkx()
    print(f"   NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
    
    # Save as GraphML for external tools
    import networkx as nx
    from pathlib import Path
    
    output_path = Path(__file__).parent / "hypergraph.graphml"
    nx.write_graphml(nx_graph, output_path)
    print(f"   Saved to {output_path}")
    
    # Step 6: Save hypergraph JSON
    json_path = Path(__file__).parent / "hypergraph.json"
    with open(json_path, 'w') as f:
        f.write(hypergraph.to_json())
    print(f"   Saved JSON to {json_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return hypergraph


def visualize_hypergraph(hypergraph):
    """
    Visualize the hypergraph using matplotlib.
    
    Note: Requires matplotlib and networkx.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        print("\n6. Visualizing hypergraph...")
        G = hypergraph.to_networkx()
        
        # Define node colors by type
        color_map = {
            'input_schema': 'lightblue',
            'output_schema': 'lightgreen',
            'effect': 'orange',
            'condition': 'yellow'
        }
        
        colors = []
        for node_id in G.nodes():
            node = hypergraph.nodes.get(UUID(node_id))
            if node:
                colors.append(color_map.get(node.node_type.value, 'gray'))
            else:
                colors.append('gray')
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, node_color=colors, with_labels=False, 
                node_size=500, arrows=True)
        
        # Add labels
        labels = {}
        for node_id in G.nodes():
            node = hypergraph.nodes.get(UUID(node_id))
            if node:
                labels[node_id] = node.name[:20] + '...' if len(node.name) > 20 else node.name
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("HyperFlow AI - Tool-Schema Hypergraph")
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "hypergraph.png", dpi=300)
        print("   Saved visualization to hypergraph.png")
        plt.show()
        
    except ImportError:
        print("   matplotlib not installed. Skipping visualization.")


if __name__ == "__main__":
    hypergraph = build_from_example_spec()
    visualize_hypergraph(hypergraph)