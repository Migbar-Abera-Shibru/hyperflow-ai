from dataclasses import Field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from uuid import UUID

from pydantic import BaseModel

from .models import ToolSchemaHypergraphm, HyperEdge, Node, NodeType

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


