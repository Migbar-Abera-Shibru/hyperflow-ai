from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set 
from uuid import UUID


from pydantic import BaseModel, Field

from .models import ToolSchemaHypergraph, HyperEdge

class AgentState(BaseModel):
    """
    Runtime agent state for execution
    
    Tracks: 
     Execution history
     subtask completion status
     available schema-value bindings 
     achieved effects

     This is used to determine what inputs ar already staisfied and
     what stilll needs to be produced
    """

    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    completed_subtasks: Set[str] = Field(default_factory=set)
    binings: Dict[str, Any] = Field(
        default_factory=dict,
        desciption = "Schema name -> value mappings"
    )
    achieved_effects: Set[str] = Field(
        default_factory=set,
        desciption = "Effects that have been achieved"
    )
    environment_state: Dict[str, Any] = Field(
        default_factory=dict,
        desciptio= "Additional environment state"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def add_binding(self, schema_name: str, value: Any) -> None: 
        # add a new schema-value binding
        self.bindings[schema_name] = value

    def add_effect(self, effect: str) -> None:
        # record that an effect has been achieved
        self.achieved_effects.add(effect)

    def add_history_entry(self, entry: Dict[str, Any]) -> None:
        # add an entry to execution history
        self.execution_history.append(entry)

    def is_available(self, schema_name: str) -> bool:
        # check if  as schema balue is available in the state, both on bindings and effects
        return (
            schema_name in self.binding or 
            schema_name in self.achieved_effects or
            schema_name in self.default_values
        )
    

    def get_values(self, schema_name: str) -> Optional[Any]:
        # get a bound value for a schema
        return self.bindings.get(schema_name)

    def has_completed(self, subtask_id: str) -> bool:
        #check if a subtask has been completed.
        return subtask_id in self.completed_subtasks


class DeficitSet(BaseModel):
    """
    A set of unresolved input requirments.

    Represents input schemas that are required by selected tools
    but are not yet available in the agent state or produced by
    the current support graph.
    
    Attributes:
        unresolved_inputs: Set of input schema node IDs
        required_by: Map from input node ID to hyperedge that requires it
        metadata: Additional information about the deficits
    """
    unresolved_inputs: Set[UUID] = Field(default_factory=set)
    required_by: Dict[UUID, Set[UUID]] = Field(default_factory=dict,
                                               description = "Input ID -> set of hyperedge Ids that require it")

    def add_deficit(
            self,
            input_node_id: UUID,
            required_by_edge: UUID
    ) -> None:
        # add a deficit with its requiring hyperedge

        self.unresolved_inputs.add(input_node_id)

        if input_node_id not in self.required_by:
            self.required_by[input_node_id] = set()
        self.required_by[input_node_id].add(required_by_edge)

    def resolve(self, input_node_id: UUID) -> None:
        # remove a deficit as resolved
        self.unresolved_inputs.discard(input_node_id)
        self.required_by.pop(input_node_id, None)

    def resolve_multiple(self, input_node_ids: Set[UUID]) -> None:
        # remove multiple deficits
        for node_id in input_node_ids:
            self.resolve(node_id)
    def is_empty(self) -> bool:
        # check if there are no unresolved deficits
        return len(self.unresolved_inputs) == 0

    def get_deficit_score(self) -> int:
        # get the number of unresolved deficits
        return len(self.unresolved_inputs)

    def to_binary_vector(self, input_node_ids: List[UUID]) -> List[int]:
        """
        Convert to binary vector for scoring. 
        This will return a list of 1s and 0s matching the input_node_ids order.
        """
        input_set = set(input_node_ids)
        return [1 if nid in self.unresolved_inputs else 0 for nid in input_set]

    def overlaps_with(self, edge: HyperEdge) -> int: 
        # calculate overlap between this deficit set and edge's outputs.
        # This is used for prioritizing producer tools during expansion.
        return sum( 1 for node_id in self.unresolved_inputs
                   if node_id in edge.output_nodes)

class SupportGraph(BaseModel):
    """
    A complete tool composition for realizing a subtask.
    
    This is the result of Deficit-Oriented Expansion:
    a set of tools, their connections, and the resolved deficits.
    
    Attributes:
        terminal_tool: The primary tool for the subtask
        producer_tools: Additional tools needed to satisfy inputs
        subgraph: The induced hypergraph subgraph
        resolved_deficits: Inputs that were satisfied by this graph
        execution_order: Topological order of tool execution
        input_requirements: Final inputs that must come from state
        output_productions: Outputs that will be produced
    """
    terminal_tool: UUID
    producer_tools: Set[UUID] = Field(default_factory=set)
    subgraph: Optional[ToolSchemaHypergraph] = None
    resolved_deficits: Set[UUID] = Field(default_factory=set)
    execution_order: List[UUID] = Field(default_factory=list)
    input_requirements: Set[UUID] = Field(default_factory=set)
    output_productions: Set[UUID] = Field(default=set)

    def get_all_tools(self) -> Set[UUID]:
        # get all tools in the support graph
        tools = {self.terminal_tool}
        tools.update(self.producer_tools)
        return tools

    def is_complete(self) -> bool:
        # check if the support graph is complete
        return len(self.producer_tools) > 0 and self.subgraph is not None

    def get_tools_in_order(self) -> List[UUID]:
        #get tools in execution order
        return self.execution_order

    def validate(self) -> List[str]:
        # validate the support graph which returns list of validation issues
        issues = []

        if not self.subgraph:
            issues.append("Subgraph is None")
            return issues

        # check all tools are in the subgraph
        all_tools = self.get_all_tools()
        missing_tools = all_tools - set(self.subgraph.hyperedges.keys())
        if missing_tools:
            issues.append(f"Missing tools in subgraph: {missing_tools}")

        # check resolved deficits are actually resolved
        for deficit_id in self.resolved_deficits:
            if deficit_id not in self.subgraph.nodes:
                issues.append(f"Resolved deficit {deficit_id} not in subgraph")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        # convert to dictionary for serialization

        return {
            'terminal_tool': str(self.terminal_tool),
            'producer_tools': [str(t) for t in self.producer_tools],
            'resolved_deficits': [str(d) for d in self.resolved_deficits],
            'execution_order': [str(t) for t in self.execution_order],
            'input_requirements': [str(i) for i in self.input_requirements],
            'output_productions': [str(o) for o in self.input_requirements]
        }

@dataclass
class SupportGraphCandidate:
    """
    A candidate support graph during beam search.
    
    Used in Deficit-Oriented Expansion to maintain
    multiple expansion paths.
    """
    tools : Set[UUID] = field(default_factory=set)
    deficits: DeficitSet = field(default_factory=DeficitSet)
    subgraph: Optional[ToolSchemaHypergraph] = None
    score: float = 0.0
    depth: int = 0

    def get_tools_ordered(self) -> List[UUID]:
        # get tools in a deterministic order
        return sorted(list(self.tools))

    def add_tool(self, tool_id: UUID) -> None:
        # add a tool to the candidate
        self.tools.add(tool_id)

    def is_complete(self) -> bool:
        # check if this candidate is complete
        return self.deficits.is_empty()

    def compare_to(self, other: 'SupportGraphCandidate') -> int:
        # compare two candidates for sorting.
        # returns -1 if this is better and 1 if the other is better and 0 if equal
        #first compare by number of defitits 
        if self.deficits.get_deficit_score() != other.deficits.get_deficit_score():
            return -1 if self.deficits.get_deficit_score() < other.deficits.get_deficit_score() else 1

        # then by score ( higher is better)
        if self.score != other.score:
            return -1 if self.score > other.score else 1

        # then by depth(the shallower the better)
        if self.depth != other.depth:
            return -1 if self.depth < other.depth else 1

        return 0
        




