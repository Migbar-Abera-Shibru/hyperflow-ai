"""
DAG Planner for HyperFlow AI.

Implements the task-level planning from the paper:
"Conditioned on this context graph, the agent is prompted to decompose 
the request into a schema-level Task DAG."

Key components:
1. Task DAG generation with LLM
2. Schema-level dependency modeling
3. Candidate tool identification
4. DAG validation and optimization
5. Incremental DAG refinement

Production considerations:
- LLM output parsing with Pydantic validation
- Token budget management
- Retry logic for malformed responses
- Caching of common task patterns
- Schema-aware validation of dependencies

The Task DAG represents a structured plan where:
- Nodes: Subtasks with specific goals and outputs
- Edges: Data/effect dependencies between subtasks
- Each subtask has candidate terminal tools
"""
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4
import networkx as nx

from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)

class Subtask(BaseModel):
    """
    A subtask in the Task DAG.
    
    Represents a unit of work that produces specific schemas
    and can be accomplished by candidate tools.
    
    Attributes:
        id: Unique subtask identifier
        goal: Natural language description of the subtask
        outputs: Schemas/effects this subtask must produce
        candidate_tools: Tools that can accomplish this subtask
        dependencies: Subtask IDs that must precede this
        status: Execution status (pending, running, completed, failed)
        support_graph: Tool support graph for execution
    """
    id: str = Field(default_factory=lambda: f"subtask_{uuid4().hex[:8]}")
    goal: str
    outputs: Set[UUID] = Field(default_factory=set)
    candidate_tools: Set[UUID] = Field(default_factory=set)
    dependecies: Set[str] = Field(default_factory=set)
    status: str = "pending"
    support_graph: Optional[Dict[str, Any]]=None
    metadata = Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @validator('dependencies')
    def no_self_dependency(cls, v: Set[str], values: Dict[str, Any]) -> Set[str]:
        # prevent self dependencies
        if 'id' in values:
            if values['id'] in v:
                raise ValueError("substack cannot depend on itself")
        return v


class TasDAG(BaseModel):
    """
    A directed acyclic graph of subtasks.
    
    Represents the complete plan for achieving a user request.
    
    Attributes:
        root_task: The original user request
        subtasks: Map of subtask ID to Subtask
        dependencies: Edges between subtasks (data/effect flow)
        context_graph: The tool context graph used for planning
        execution_order: Topological order of execution
        status: Overall DAG status
    """
    root_task: str
    subtasks: Dict[str, Subtask] = Field(default_factory=dict)
    dependencies: List[Tuple[str, str]] = Field(default_factory=list)
    context_graph_id: Optional[str] = None
    execution_order: List[str] = Field(default_factory=list)
    status: str = "planned"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def get_substack(self, subtask_id: str) -> Optional[Subtask]:
        # get a subtask by Id
        return self.subtasks.get(subtask_id)

    def get_ready_subtasks(self) -> List[Subtask]:
        """
        Get subtasks whose dependencies are satisfied.
        
        Returns subtasks that can be executed now.
        """
        ready = []
        completed = {
            sid for sid, subtask in self.subtasks.items()
            if subtask.status == "completed"
        }

        for subtask in self.subtasks.values():
            if subtask.status != "pending":
                continue

            # check if all dependencies are completed
            if subtask.dependencies.issubset(completed):
                ready.append(subtask)

        return ready

    def get_completed_count(self) -> int:
        # get number of completed subtasks
        return sum(1 for s in self.subtasks.values() if s.status == "completed")

    def get_failed_count(self) -> int:
        # get number of failed subtasks
        return sum(1 for s in self.subtasks.values() if s.status == "failed")

    def is_complete(self) -> bool:
        # check if all subtasks are completed or failed
        terminal_statuses = {"completed", "failed", "skipped"}
        return all(s.status in terminal_statuses for s in self.subtasks.values())

    def is_successful(self) -> bool:
        # get completion progress as a percentage
        if not self.subtasks:
            return 0.0
        total = len(self.subtasks)
        completed = self.get_completed_count()
        return (completed / total) * 100
    
    def topological_sort(self) -> List[str]:
        # get subtasks in topological order
        if self.execution_order:
            return self.execution_order

        # build graph for topolgical sort
        G = nx.DiGraph()

        # add nodes
        for subtask_id in self.subtasks:
            G.add_node(subtask_id)

        # add edges
        for source, target in self.dependencies:
            G.add_edge(source, target)

        # check for cycles
        try:
            order = list(nx.topological_sort(G))
            self.execution_order = order
            return order
        except nx.NetworkXUnfeasible:
            logger.error("DAG has cycles, cannot sort")
            # try to find and break cycles
            cycles = list(nx.simple_cycles(G))
            logger.warning(f"Found Cycles: {cycles}")
            # return in original order
            return list(self.subtasks.keys())

    def to_json(self) -> str:
        # serialize DAG to JSON
        return self.model_dump_json(indent=2)

    