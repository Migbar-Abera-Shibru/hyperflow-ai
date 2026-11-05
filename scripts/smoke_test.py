"""
Smoke test - verifies all modules can be imported.
Run this FIRST before any functional tests.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("SMOKE TEST: Importing all modules")
    print("=" * 60)
    
    # Test 1: Core models
    try:
        from hyperflow.core.models import (
            Node, NodeType, HyperEdge, Dependency,
            ToolSchemaHypergraph, SupportMatrix
        )
        print("✅ core.models imported")
    except Exception as e:
        print(f"❌ core.models FAILED: {e}")
        return False
    
    # Test 2: Core support
    try:
        from hyperflow.core.support import (
            AgentState, DeficitSet, SupportGraph, SupportGraphCandidate
        )
        print("✅ core.support imported")
    except Exception as e:
        print(f"❌ core.support FAILED: {e}")
        return False
    
    # Test 3: OpenAPI parser
    try:
        from hyperflow.builders.open_api_parser import (
            OpenAPIParser, ToolDefinition, ParameterSchema
        )
        print("✅ builders.openapi_parser imported")
    except Exception as e:
        print(f"❌ builders.openapi_parser FAILED: {e}")
        return False
    
    # Test 4: Hypergraph builder
    try:
        from hyperflow.builders.hypergraph_builder import HypergraphBuilder
        print("✅ builders.hypergraph_builder imported")
    except Exception as e:
        print(f"❌ builders.hypergraph_builder FAILED: {e}")
        return False
    
    # Test 5: Embeddings
    try:
        from hyperflow.utils.embeddings import (
            EmbeddingManager, EmbeddingConfig, SimilarityScorer
        )
        print("✅ utils.embeddings imported")
    except Exception as e:
        print(f"❌ utils.embeddings FAILED: {e}")
        return False
    
    # Test 6: Context extractor
    try:
        from hyperflow.planning.context_extractor import (
            ContextExtractor, ContextExtractionConfig, TaskInterpretation
        )
        print("✅ planning.context_extractor imported")
    except Exception as e:
        print(f"❌ planning.context_extractor FAILED: {e}")
        return False
    
    # Test 7: DAG planner
    try:
        from hyperflow.planning.dag_planner import DAGPlanner, TaskDAG, Subtask
        print("✅ planning.dag_planner imported")
    except Exception as e:
        print(f"❌ planning.dag_planner FAILED: {e}")
        return False
    
    # Test 8: DOE
    try:
        from hyperflow.planning.doe import DeficitOrientedExpansion, DOEConfig
        print("✅ planning.doe imported")
    except Exception as e:
        print(f"❌ planning.doe FAILED: {e}")
        return False
    
    # Test 9: Execution engine
    try:
        from hyperflow.execution.engine import (
            ExecutionEngine, SubtaskExecutionResult, ToolExecutionResult
        )
        print("✅ execution.engine imported")
    except Exception as e:
        print(f"❌ execution.engine FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL IMPORTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)