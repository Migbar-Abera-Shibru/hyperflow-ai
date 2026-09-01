"""  """


from uuid import uuid4

import numpy as np
import pytest

from hyperflow.core.models import Dependency, HyperEdge, Node, NodeType, SupportMatrix, ToolSchemaHypergraph


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
                node_type=NodeType.INPUT_SCHEMA,
                description="" # if descirption is empty
            )

    def test_node_to_network(slef):
        node = Node(
            name="test_node",
            node_type=NodeType.OUTPUT_SCHEMA,
            description="Test output"
        )

        nx_attrs = node.to_network_node()
        assert nx_attrs['name'] == "test_node"
        assert nx_attrs['type'] == "output_schema"
        assert nx_attrs['description'] == "Test output"

class TestHyperEdge:
    """ Test for hyperedge model"""
    def test_hyperedge_creation(self):
        """ Test the basic hyperedge creation"""
        input_node = Node(
            name="input_1",
            node_type=NodeType.INPUT_SCHEMA,
            description="Input1"
        )

        output_node= Node (
            name="output_1",
            node_type=NodeType.OUTPUT_SCHEMA,
            description="output 1"
        )

        edge = HyperEdge(
            name = "test_tool",
            description="A test tool",
            input_node={input_node.id},
            output_node={output_node.id}
        )

        assert edge.name == "test_tool"
        assert input_node.id in edge.input_nodes
        assert output_node.id in edge.output_nodes

    def test_hyperedge_validation(self):
        """ Test the validation of hyperedge"""
        with pytest.raises(ValueError):
            HyperEdge(
                name = "test_tool",
                description="test",
                input_nodes=set(), # no inputs
                output_nodes={uuid4()}
            )
        with pytest.raises(ValueError):
            HyperEdge(
                name="test_tool",
                description="test",
                input_nodes={uuid4()},
                output_nodes=set() # for no outputs
            )

    def test_hyperedge_getters(self):
        """ Test input/output getters"""
        input_ids = {uuid4(), uuid4()}
        output_ids = {uuid4()}

        edge = HyperEdge(
            name="test",
            description="test",
            input_nodes=input_ids,
            output_nodes=output_ids
        )

        assert set(edge.get_inputs()) == input_ids
        assert set(edge.get_outputs()) == output_ids

class TestToolSchemaHypergraph:
    """ Test for  ToolSchemaHypergraph"""
    def setup_method(self):
        self.hypergraph = ToolSchemaHypergraph

        #creat nodes
        self.input1 = Node(
            name = "user_id",
            node_type=NodeType.INPUT_SCHEMA,
            description="User id"
        )
        self.input2 = Node(
            name="amount",
            node_type=NodeType.INPUT_SCHEMA,
            description="Amount"
        )
        self.output1 = Node(
            name="transaction_id",
            node_type=NodeType.OUTPUT_SCHEMA,
            description="Transactin ID"
        )

        self.effect1 = Node(
            name="payment_sent",
            node_type=NodeType.EFFECT,
            description="Payment was sent"
        )
        self.hypergraph.add_node(self.input1)
        self.hypergraph.add_node(self.input2)
        self.hypergraph.add_node(self.output1)
        self.hypergraph.add_node(self.effect1)

        #create hyperedge
        self.payment_tool = HyperEdge(
            name="send_payment",
            description="Send a payment",
            input_nodes={self.input1.id, self.input2.id},
            output_nodes={self.output1.id, self.effect1.id}
        )

        self.hypergraph.add_hyperedge(self.payment_tool)

    def test_add_node(self):
        """ Test adding nodes"""
        new_node = Node(
            name="currency",
            node_type=NodeType.INPUT_SCHEMA,
            description="Currency code"
        )

        self.hypergraph.add_node(new_node)
        assert self.hypergraph.get_node_by_name("currency") == new_node

    def test_add_hyperedge(self):
        new_edge = HyperEdge(
            name="get_user",
            description="Get user info",
            input_nodes=set(),
            output_nodes={self.output1.id}
        )

        self.hypergraph.add_hyperedge(new_edge)
        assert self.hypergraph.get_hyperedge_by_name("get_user") == new_edge

    def test_add_dependency(self):
        """ Test adding dependency"""
        dep = Dependency(
            source_node=self.output1.id,
            target_node=self.input1.id,
            weight=0.9
        )
        self.hypergraph.add_dependency(dep)
        assert dep.id in self.hypergraph.dependencies

    def test_get_producers_for_input(self):
        """Test getting the producer tools """
        # add dependecy from output to input

        dep = Dependency(
            source_node= self.output1.id,
            target_node=self.input1.id,
            weight=0.9
        )
        self.hypergraph.add_dependency(dep)

        producers = self.hypergraph.get_producers_for_input(self.input1.id)
        assert len(producers) == 1
        assert producers[0][0] == self.payment_tool.id

    def test_build_support_matrix(self):
        """ test for support matrix building"""
        dep = Dependency(source_node=self.output1.id,
                         target_node=self.input1.id,
                         weight=0.9)
        self.hypergraph.add_dependency(dep)

        matrix = self.hypergraph.build_support_matrix()

        #should have two input node and 1 hyperedge 
        assert matrix.shape == (2,1)

        # check the support scores
        assert matrix[0,0] == 0.9 # input1 support from payment_tool
        assert matrix[1,0] == 0.0 # input2 has no support

    def test_get_subgraph(self):
        """ Test the subgraph extraction"""
        # create a second tool

        input3 = Node(
            name="user_name",
            node_type=NodeType.INPUT_SCHEMA,
            description="User name"
        )
        self.hypergraph.add_node(input3)

        second_tool = HyperEdge(
            name="get_user_info",
            description="Get user information",
            input_nodes = {input3.id},
            output_nodes={self.input1.id}
        )
        self.hypergraph.add_hyperedge(second_tool)

        # extract the subgraph with just the first tool
        subgraph = self.hypergraph.get_subgraph({self.payment_tool.id})

        # should only contain nodes from the payment tool
        expectedd_nodes = {self.input1.id, self.input2.id,
                         self.output1.id, self.effect1.id}
        assert set(subgraph.nodes.keys()) == expectedd_nodes

    def test_validate(self):
        """test the hypergraph validation"""
        issues = self.hypergraph.validate()
        assert len(issues) == 0

        # add invalid dependency
        invalid_node = uuid4()
        dep = Dependency(
            source_node = invalid_node,
            target_node = self.input1.id,
            weight=0.5
        )
        self.hypergraph.add_dependency(dep)

        issues = self.hypergraph.validate()
        assert len(issues) > 0

    def test_serialization(self):
        """ test JSON serialization/deserialization"""
        json_str = self.hypergraph.to_json()
        loaded = ToolSchemaHypergraph.from_json(json_str)

        # should have the same structure
        assert len(loaded.nodes) == len(self.hypergraph.nodes)
        assert len(loaded.hyperedges) == len(self.hypergraph.hyperedges)


class TestSupportMatrix:
    """ Tests for SupportMatrix"""

    def test_support_matrix_lookup(self):
        """ test support matrix lookups"""
        input_ids = [uuid4(), uuid4()]
        edge_ids = [uuid4(), uuid4()]

        matrix = np.array([
            [0.9, 0.0],
            [0.5, 0.0]
        ])

        support = SupportMatrix(matrix, input_ids, edge_ids)

        # test get_producers
        producers = support.get_producers(input_ids[0])
        assert producers[0][0] == edge_ids[0]
        assert producers[0][1] == 0.9


        # test get_top_producers
        top = support.get_top_producers(input_ids[1], k = 1)
        assert top[0][0] == edge_ids[1]
        assert top[0][1] == 0.8

        # test get_support_score
        score = support.get_support_score(edge_ids[0], input_ids[0])
        assert score == 0.9



    