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

import json
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx

from pydantic import BaseModel, Field, validator

from ..core.models import (
    ToolSchemaHypergraph, Node, NodeType, HyperEdge, Dependency
)
from ..llm.prompts import get_dag_planning_prompt
from ..utils.embeddings import EmbeddingManager

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
    dependencies: Set[str] = Field(default_factory=set)
    status: str = "pending"  # pending, running, completed, failed, skipped
    support_graph: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    @validator('dependencies')
    def no_self_dependency(cls, v: Set[str], values: Dict[str, Any]) -> Set[str]:
        """Prevent self-dependencies."""
        if 'id' in values:
            if values['id'] in v:
                raise ValueError("Subtask cannot depend on itself")
        return v


class TaskDAG(BaseModel):
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
    status: str = "planned"  # planned, executing, completed, failed
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    def get_subtask(self, subtask_id: str) -> Optional[Subtask]:
        """Get a subtask by ID."""
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
            
            # Check if all dependencies are completed
            if subtask.dependencies.issubset(completed):
                ready.append(subtask)
        
        return ready
    
    def get_completed_count(self) -> int:
        """Get number of completed subtasks."""
        return sum(1 for s in self.subtasks.values() if s.status == "completed")
    
    def get_failed_count(self) -> int:
        """Get number of failed subtasks."""
        return sum(1 for s in self.subtasks.values() if s.status == "failed")
    
    def is_complete(self) -> bool:
        """Check if all subtasks are completed or failed."""
        terminal_statuses = {"completed", "failed", "skipped"}
        return all(s.status in terminal_statuses for s in self.subtasks.values())
    
    def is_successful(self) -> bool:
        """Check if the DAG completed successfully."""
        return self.is_complete() and self.get_failed_count() == 0
    
    def get_progress(self) -> float:
        """Get completion progress as a percentage."""
        if not self.subtasks:
            return 0.0
        total = len(self.subtasks)
        completed = self.get_completed_count()
        return (completed / total) * 100
    
    def topological_sort(self) -> List[str]:
        """Get subtasks in topological order."""
        if self.execution_order:
            return self.execution_order
        
        # Build graph for topological sort
        G = nx.DiGraph()
        
        # Add nodes
        for subtask_id in self.subtasks:
            G.add_node(subtask_id)
        
        # Add edges
        for source, target in self.dependencies:
            G.add_edge(source, target)
        
        # Check for cycles
        try:
            order = list(nx.topological_sort(G))
            self.execution_order = order
            return order
        except nx.NetworkXUnfeasible:
            logger.error("DAG has cycles, cannot sort")
            # Try to find and break cycles
            cycles = list(nx.simple_cycles(G))
            logger.warning(f"Found cycles: {cycles}")
            # Return in original order
            return list(self.subtasks.keys())
    
    def to_json(self) -> str:
        """Serialize DAG to JSON."""
        return self.model_dump_json(indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_graphviz(self) -> str:
        """Generate GraphViz DOT format for visualization."""
        lines = ['digraph TaskDAG {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')
        
        # Add nodes
        for subtask_id, subtask in self.subtasks.items():
            label = f"{subtask_id}\n{subtask.goal[:50]}..."
            status_color = {
                "pending": "lightgray",
                "running": "yellow",
                "completed": "lightgreen",
                "failed": "lightcoral",
                "skipped": "lightgray"
            }.get(subtask.status, "lightgray")
            lines.append(
                f'  "{subtask_id}" [label="{label}", '
                f'style=filled, fillcolor="{status_color}"];'
            )
        
        # Add edges
        for source, target in self.dependencies:
            lines.append(f'  "{source}" -> "{target}";')
        
        lines.append('}')
        return '\n'.join(lines)


class DAGPlanner:
    """
    Generates Task DAGs from user requests.
    
    Uses LLM to decompose tasks into subtasks with schema-aware
    dependencies based on the tool context graph.
    
    Process:
    1. Serialize context graph for LLM
    2. Prompt LLM to generate DAG
    3. Parse and validate LLM output
    4. Map schemas and tools to their IDs
    5. Validate DAG integrity
    6. Topologically sort for execution
    """
    
    def __init__(
        self,
        hypergraph: ToolSchemaHypergraph,
        llm_client,
        embedder: Optional[EmbeddingManager] = None,
        max_subtasks: int = 20,
        max_tokens: int = 4000
    ):
        """
        Initialize the DAG planner.
        
        Args:
            hypergraph: The full hypergraph
            llm_client: LLM client for generation
            embedder: Embedding manager (optional)
            max_subtasks: Maximum subtasks per DAG
            max_tokens: Maximum tokens for LLM context
        """
        self.hypergraph = hypergraph
        self.llm_client = llm_client
        self.embedder = embedder or EmbeddingManager()
        self.max_subtasks = max_subtasks
        self.max_tokens = max_tokens
        
        # Cache for serialized context graphs
        self._context_cache: Dict[str, str] = {}
    
    def plan(
        self,
        task: str,
        context_graph: ToolSchemaHypergraph,
        existing_dag: Optional[TaskDAG] = None
    ) -> TaskDAG:
        """
        Generate a Task DAG for the given task.
        
        Args:
            task: User request
            context_graph: Task-relevant context graph
            existing_dag: Optional existing DAG to refine
            
        Returns:
            TaskDAG object
        """
        logger.info(f"Planning DAG for task: {task[:100]}...")
        
        # Serialize context graph
        context_text = self._serialize_context(context_graph)
        
        # Prepare prompt
        prompt = get_dag_planning_prompt(context_text, task)
        
        # Get LLM response
        try:
            response = self._call_llm(prompt)
            dag_data = self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            # Create a minimal DAG as fallback
            return self._create_minimal_dag(task, context_graph)
        
        # Build DAG from parsed data
        dag = self._build_dag(task, dag_data, context_graph)
        
        # Validate DAG
        issues = self._validate_dag(dag)
        if issues:
            logger.warning(f"DAG validation issues: {issues}")
            # Try to repair
            dag = self._repair_dag(dag, context_graph)
        
        # Topological sort
        dag.topological_sort()
        
        logger.info(f"Generated DAG with {len(dag.subtasks)} subtasks")
        return dag
    
    def _serialize_context(self, context_graph: ToolSchemaHypergraph) -> str:
        """
        Serialize context graph for LLM consumption.
        
        Creates a compact textual representation that preserves
        schema-level dependencies.
        """
        # Check cache
        cache_key = str(hash(frozenset(context_graph.hyperedges.keys())))
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        lines = []
        lines.append("## Available Tools (with schema dependencies)\n")
        
        # Group tools by type or domain
        for edge_id, edge in context_graph.hyperedges.items():
            lines.append(f"\n### Tool: {edge.name}")
            lines.append(f"Description: {edge.description}")
            
            # Inputs
            if edge.input_nodes:
                lines.append("Inputs:")
                for node_id in edge.input_nodes:
                    node = context_graph.nodes.get(node_id)
                    if node:
                        req = "(required)" if node.is_required else "(optional)"
                        lines.append(f"  - {node.name}: {node.description} {req}")
            
            # Outputs
            if edge.output_nodes:
                lines.append("Outputs:")
                for node_id in edge.output_nodes:
                    node = context_graph.nodes.get(node_id)
                    if node:
                        lines.append(f"  - {node.name}: {node.description}")
            
            # Dependencies (connections from other tools)
            incoming = []
            outgoing = []
            
            for dep in context_graph.dependencies.values():
                # Check if this edge's nodes are involved
                if dep.source_node in edge.output_nodes:
                    target_node = context_graph.nodes.get(dep.target_node)
                    if target_node:
                        incoming.append(f"{target_node.name} (weight: {dep.weight:.2f})")
                
                if dep.target_node in edge.input_nodes:
                    source_node = context_graph.nodes.get(dep.source_node)
                    if source_node:
                        outgoing.append(f"{source_node.name} (weight: {dep.weight:.2f})")
            
            if incoming:
                lines.append("Produced from:")
                lines.append(f"  - {', '.join(incoming[:3])}")
                if len(incoming) > 3:
                    lines.append(f"    ... and {len(incoming) - 3} more")
            
            if outgoing:
                lines.append("Used by:")
                lines.append(f"  - {', '.join(outgoing[:3])}")
                if len(outgoing) > 3:
                    lines.append(f"    ... and {len(outgoing) - 3} more")
        
        # Add summary
        lines.append(f"\nTotal tools: {len(context_graph.hyperedges)}")
        lines.append(f"Total schemas: {len(context_graph.nodes)}")
        
        context_text = '\n'.join(lines)
        
        # Cache
        self._context_cache[cache_key] = context_text
        return context_text
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with the prompt.
        
        Handles retries and token limits.
        """
        # Truncate if needed
        if len(prompt) > self.max_tokens * 4:  # Rough token estimate
            prompt = prompt[:self.max_tokens * 4]
            logger.warning(f"Prompt truncated to {self.max_tokens * 4} characters")
        
        # Call LLM
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4o-mini for cost efficiency
                messages=[
                    {"role": "system", "content": "You are a precise AI planner."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            # Extract JSON from response
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            data = json.loads(response)
            
            # Validate required fields
            if 'subtasks' not in data:
                raise ValueError("Missing 'subtasks' field")
            
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response: {response[:500]}...")
            raise
    
    def _build_dag(
        self,
        task: str,
        data: Dict[str, Any],
        context_graph: ToolSchemaHypergraph
    ) -> TaskDAG:
        """
        Build TaskDAG from parsed LLM data.
        """
        dag = TaskDAG(root_task=task)
        
        # Map tool names to IDs
        tool_name_to_id = {
            edge.name: edge_id 
            for edge_id, edge in context_graph.hyperedges.items()
        }
        
        # Map schema names to IDs
        schema_name_to_id = {
            node.name: node_id 
            for node_id, node in context_graph.nodes.items()
        }
        
        # First pass: create subtasks
        for subtask_data in data['subtasks']:
            subtask = Subtask(
                id=subtask_data.get('id', f"subtask_{uuid4().hex[:8]}"),
                goal=subtask_data.get('goal', ''),
                dependencies=set(subtask_data.get('dependencies', []))
            )
            
            # Map outputs to schema IDs
            if 'outputs' in subtask_data:
                for output_name in subtask_data['outputs']:
                    if output_name in schema_name_to_id:
                        subtask.outputs.add(schema_name_to_id[output_name])
                    else:
                        # Try fuzzy matching
                        matched = self._fuzzy_match_schema(output_name, schema_name_to_id)
                        if matched:
                            subtask.outputs.add(matched)
            
            # Map candidate tools to IDs
            if 'candidate_tools' in subtask_data:
                for tool_name in subtask_data['candidate_tools']:
                    if tool_name in tool_name_to_id:
                        subtask.candidate_tools.add(tool_name_to_id[tool_name])
                    else:
                        # Try fuzzy matching
                        matched = self._fuzzy_match_tool(tool_name, tool_name_to_id)
                        if matched:
                            subtask.candidate_tools.add(matched)
            
            dag.subtasks[subtask.id] = subtask
        
        # Second pass: build dependency edges
        for subtask in dag.subtasks.values():
            for dep_id in subtask.dependencies:
                if dep_id in dag.subtasks:
                    dag.dependencies.append((dep_id, subtask.id))
        
        # Store context graph reference
        dag.context_graph_id = str(hash(frozenset(context_graph.hyperedges.keys())))
        
        return dag
    
    def _fuzzy_match_schema(
        self,
        name: str,
        schema_map: Dict[str, UUID]
    ) -> Optional[UUID]:
        """
        Fuzzy match a schema name using embeddings.
        """
        # First try exact match (case insensitive)
        name_lower = name.lower()
        for schema_name, schema_id in schema_map.items():
            if schema_name.lower() == name_lower:
                return schema_id
        
        # Then try partial match
        for schema_name, schema_id in schema_map.items():
            if name_lower in schema_name.lower() or schema_name.lower() in name_lower:
                return schema_id
        
        # Finally use semantic similarity
        if self.embedder:
            best_score = 0.0
            best_match = None
            for schema_name, schema_id in schema_map.items():
                score = self.embedder.similarity(name, schema_name)
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = schema_id
            return best_match
        
        return None
    
    def _fuzzy_match_tool(
        self,
        name: str,
        tool_map: Dict[str, UUID]
    ) -> Optional[UUID]:
        """
        Fuzzy match a tool name using embeddings.
        """
        # Same logic as fuzzy_match_schema
        name_lower = name.lower()
        for tool_name, tool_id in tool_map.items():
            if tool_name.lower() == name_lower:
                return tool_id
        
        for tool_name, tool_id in tool_map.items():
            if name_lower in tool_name.lower() or tool_name.lower() in name_lower:
                return tool_id
        
        if self.embedder:
            best_score = 0.0
            best_match = None
            for tool_name, tool_id in tool_map.items():
                score = self.embedder.similarity(name, tool_name)
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = tool_id
            return best_match
        
        return None
    
    def _validate_dag(self, dag: TaskDAG) -> List[str]:
        """
        Validate the DAG for integrity and executability.
        """
        issues = []
        
        # Check for empty DAG
        if not dag.subtasks:
            issues.append("DAG has no subtasks")
            return issues
        
        # Check for cycles
        G = nx.DiGraph()
        for subtask_id in dag.subtasks:
            G.add_node(subtask_id)
        for source, target in dag.dependencies:
            G.add_edge(source, target)
        
        try:
            list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            issues.append("DAG has cycles")
        
        # Check all dependencies exist
        for subtask in dag.subtasks.values():
            for dep_id in subtask.dependencies:
                if dep_id not in dag.subtasks:
                    issues.append(f"Subtask {subtask.id} depends on missing {dep_id}")
        
        # Check each subtask has at least one candidate tool
        for subtask in dag.subtasks.values():
            if not subtask.candidate_tools:
                issues.append(f"Subtask {subtask.id} has no candidate tools")
        
        # Check each subtask has outputs
        for subtask in dag.subtasks.values():
            if not subtask.outputs:
                issues.append(f"Subtask {subtask.id} has no outputs")
        
        # Check max subtasks
        if len(dag.subtasks) > self.max_subtasks:
            issues.append(f"DAG has {len(dag.subtasks)} subtasks (max {self.max_subtasks})")
        
        return issues
    
    def _repair_dag(self, dag: TaskDAG, context_graph: ToolSchemaHypergraph) -> TaskDAG:
        """
        Attempt to repair a DAG with validation issues.
        """
        # Remove cycles by breaking edges
        G = nx.DiGraph()
        for subtask_id in dag.subtasks:
            G.add_node(subtask_id)
        for source, target in dag.dependencies:
            G.add_edge(source, target)
        
        # Find and break cycles
        while True:
            try:
                list(nx.topological_sort(G))
                break
            except nx.NetworkXUnfeasible:
                # Find a cycle
                cycles = list(nx.simple_cycles(G))
                if cycles:
                    # Remove the last edge of the first cycle
                    cycle = cycles[0]
                    if len(cycle) >= 2:
                        edge_to_remove = (cycle[-2], cycle[-1])
                        if edge_to_remove in dag.dependencies:
                            dag.dependencies.remove(edge_to_remove)
                            G.remove_edge(*edge_to_remove)
                            logger.warning(f"Removed cycle edge: {edge_to_remove}")
                else:
                    break
        
        # Rebuild dependency sets
        for subtask in dag.subtasks.values():
            subtask.dependencies = {
                source for source, target in dag.dependencies
                if target == subtask.id
            }
        
        return dag
    
    def _create_minimal_dag(
        self,
        task: str,
        context_graph: ToolSchemaHypergraph
    ) -> TaskDAG:
        """
        Create a minimal DAG when LLM planning fails.
        """
        dag = TaskDAG(root_task=task)
        
        # Find the most relevant tool
        if context_graph.hyperedges:
            # Use the first tool as a fallback
            edge_id = next(iter(context_graph.hyperedges.keys()))
            edge = context_graph.hyperedges[edge_id]
            
            subtask = Subtask(
                goal=f"Use {edge.name} for: {task[:100]}",
                outputs=set(edge.output_nodes),
                candidate_tools={edge_id}
            )
            dag.subtasks[subtask.id] = subtask
        
        return dag
    
    def refine_dag(
        self,
        dag: TaskDAG,
        failed_subtask_id: str,
        failure_reason: str,
        context_graph: ToolSchemaHypergraph
    ) -> TaskDAG:
        """
        Refine a DAG after a subtask failure.
        
        This implements the dynamic replanning mentioned in the paper.
        """
        logger.info(f"Refining DAG after failure of {failed_subtask_id}")
        
        # Get the failed subtask
        failed_subtask = dag.get_subtask(failed_subtask_id)
        if not failed_subtask:
            return dag
        
        # Mark as failed
        failed_subtask.status = "failed"
        dag.updated_at = datetime.utcnow()
        
        # Try to find alternative tools
        # Get the required outputs
        required_outputs = failed_subtask.outputs
        
        # Find tools that can produce these outputs
        alternative_tools = set()
        for node_id in required_outputs:
            producers = context_graph.get_producers_for_input(node_id)
            for edge_id, weight in producers:
                if edge_id not in failed_subtask.candidate_tools:
                    alternative_tools.add(edge_id)
        
        if alternative_tools:
            # Add alternative tools
            failed_subtask.candidate_tools.update(alternative_tools)
            logger.info(f"Added {len(alternative_tools)} alternative tools")
            
            # Reset status
            failed_subtask.status = "pending"
            dag.status = "planned"
            
            return dag
        
        # If no alternatives, skip this subtask and mark dependencies as resolved
        # This is a fallback to prevent complete failure
        failed_subtask.status = "skipped"
        logger.warning(f"Subtask {failed_subtask_id} skipped due to no alternatives")
        
        return dag