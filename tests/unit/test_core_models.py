"""  """


import pytest

from hyperflow.core.models import Node, NodeType


class TestNode:
    """ Tests for node model"""
    def test_node_creation(self):
        node = Node(
            name =  "user_id",
            node_type=NodeType.INPUT_SCHEMA,
            description="User identifier"

        )

        assert node.name == "user_id"
        assert node.node_type == NodeType.INPUT_SCHEMA
        assert node.is_required is True
        assert node.id is not None

    def test_node_validation(self):
        """Test node validation"""
        with pytest.raises(ValueError):
            Node(
                name = "", #if empty name
                node_type=NodeType.INPUT_SCHEMA,
                description="" # if empty description

            )
        with pytest.raises(ValueError):
            Node(
                name="test",
                node_type=NodeType.INPUT_SCHEMA
            )