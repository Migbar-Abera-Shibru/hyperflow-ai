"""
Prompt templates for HyperFlow AI.

Contains carefully engineered prompts for:
1. Task Interpretation - Structuring user requests
2. DAG Planning - Decomposing tasks into subtasks
3. Deficit Resolution - Identifying missing inputs
4. Subtask Verification - Checking completion

These prompts are designed for:
- Consistent output format (JSON schema)
- Schema-aware reasoning
- Cost optimization (concise but complete)

Production considerations:
- Versioning for prompt iteration
- Few-shot examples for consistency
- Output validation with Pydantic
- Token usage tracking
"""

class PromptTemplates:
    """ Collection of prompt templates"""

    # prompt for task interpretation
    TASK_INTERPRETATION = """
You are a task analysis expert. Given a user request, extract:
1. The core operations needed
2. The schemas/entities involved
3. Any constraints
4. Desired outcomes

User Request: {task}

Output as JSON:
{{
    "operations": "description of what operations are needed",
    "schemas": "description of the schemas/entities involved", 
    "entities": ["list of key entities"],
    "constraints": ["list of constraints"],
    "desired_outcomes": ["list of desired outcomes"]
}}

Be specific about the data that needs to be produced or consumed.
"""

    # prompt for dag planning
    # DAG Planning Prompt
    DAG_PLANNING = """
You are an AI agent planner. Given a task and available tools (as a hypergraph context),
decompose the task into a directed acyclic graph of subtasks.

Available Tools Context:
{context_graph}

Task: {task}

For each subtask, specify:
- Its goal
- The output schemas it must produce
- Candidate terminal tools that can achieve it
- Dependencies on other subtasks

Output as JSON:
{{
    "subtasks": [
        {{
            "id": "subtask_1",
            "goal": "description of what this subtask accomplishes",
            "outputs": ["schema1", "schema2"],
            "candidate_tools": ["tool1", "tool2"],
            "dependencies": []  # IDs of subtasks that must precede this
        }}
    ]
}}

Rules:
1. Subtasks must be acyclic
2. Each subtask should have at least one candidate tool
3. Dependencies should reflect data/effect flow
4. The DAG should be executable (all inputs available from previous subtasks)

Think about data dependencies, not just ordering.
"""
    # prompt for deficit resolution
    DEFICIT_RESOLUTION = """
You are an AI planning assistant. Given a subtask, its candidate tools,
and the current agent state, identify missing inputs that need to be produced.

Subtask: {subtask_goal}
Candidate Tools: {candidate_tools}
Available State: {available_state}

For each candidate tool, identify:
1. Which of its required inputs are available
2. Which inputs are missing (deficits)
3. What schemas need to be produced to resolve deficits

Output as JSON:
{{
    "tool_analyses": [
        {{
            "tool": "tool_name",
            "available_inputs": ["input1", "input2"],
            "deficits": ["input3", "input4"],
            "needed_producers": ["producer1", "producer2"]
        }}
    ],
    "overall_deficits": ["list of all missing inputs"],
    "resolution_plan": "brief description of how to resolve deficits"
}}

Focus on schema-level requirements, not just tool names.
"""

    # prompt for subtask verification
    SUBTASK_VERIFICATION = """
You are an AI execution validator. Given the execution of a subtask,
determine if it was completed successfully.

Subtask: {subtask_goal}
Expected Outputs: {expected_outputs}
Execution Trace: {execution_trace}
Environment State: {environment_state}

Check:
1. Were all expected outputs produced?
2. Are the outputs valid (correct types, values)?
3. Were the intended effects achieved?
4. Are there any errors or anomalies?

Output as JSON:
{{
    "success": true/false,
    "produced_outputs": ["list of produced schemas"],
    "missing_outputs": ["list of expected outputs not produced"],
    "errors": ["list of errors encountered"],
    "verification_details": "detailed explanation of the verification"
}}

Be thorough but concise. Verify based on evidence, not assumptions.
"""

    # replanning prompt
    REPLANNING = """
You are an AI agent replanner. A subtask execution failed or produced
unexpected results. Adjust the DAG accordingly.

Original DAG: {original_dag}
Failed Subtask: {failed_subtask}
Failure Reason: {failure_reason}
Current State: {current_state}

Determine:
1. Should the failed subtask be retried with different tools?
2. Can the subtask be modified to work with current state?
3. Should the DAG be restructured?
4. Are there alternative paths to the goal?

Output as JSON:
{{
    "action": "retry|modify|restructure|abort",
    "updated_dag": {updated_dag_json},
    "changes": ["list of changes made"],
    "reasoning": "explanation of decisions"
}}

Be pragmatic. If a subtask fails, the system should still try to complete the overall goal.
"""

    # few shot examples
    FEW_SHOT_EXAMPLES = """
Example 1:
Task: "Send $50 to Alice and notify Bob about the payment"
Subtask DAG:
[
    {{"id": "subtask_1", "goal": "Get Alice's user ID", "outputs": ["user_id"], 
      "candidate_tools": ["get_user_by_email"], "dependencies": []}},
    {{"id": "subtask_2", "goal": "Send payment to Alice", "outputs": ["transaction_id", "payment_status"],
      "candidate_tools": ["send_payment"], "dependencies": ["subtask_1"]}},
    {{"id": "subtask_3", "goal": "Notify Bob about payment", "outputs": ["notification_sent"],
      "candidate_tools": ["send_message"], "dependencies": ["subtask_2"]}}
]

Example 2:
Task: "Create a new project and add team members"
Subtask DAG:
[
    {{"id": "subtask_1", "goal": "Create project", "outputs": ["project_id"],
      "candidate_tools": ["create_project"], "dependencies": []}},
    {{"id": "subtask_2", "goal": "Get team members list", "outputs": ["members"],
      "candidate_tools": ["list_team_members"], "dependencies": []}},
    {{"id": "subtask_3", "goal": "Add members to project", "outputs": ["project_members_added"],
      "candidate_tools": ["add_project_members"], "dependencies": ["subtask_1", "subtask_2"]}}
]
"""

def get_task_interpretation_prompt(task: str) -> str:
    # get the task inperpretation prompt for specific task
    return PromptTemplates.TASK_INTERPRETATION.format(task=task)

def get_dag_planning_prompt(context_graph: str, task: str) -> str:
    # get the dag planning prompt with context
    # limit context size to avoid token overflow
    context_preview = context_graph[:4000] if len(context_graph) > 4000 else context_graph
    return PromptTemplates.DAG_PLANNING.format(
        context_graph=context_preview,
        task=task
    )

def get_deficit_resolution_prompt(
        subtask_goal: str,
        candidate_tools: str,
        available_state: str
) -> str:
    # get the deficit resolution prompt
    return PromptTemplates.DEFICIT_RESOLUTION.format(
        subtask_goal=subtask_goal,
        candidate_tools=candidate_tools,
        available_state=available_state
    )

def get_subtask_verification_prompt(
        subtask_goal: str,
        expected_outputs: str,
        execution_trace: str,
        environment_state: str,
) -> str:
    # get the subtask verification prompt
    return PromptTemplates.SUBTASK_VERIFICATION.format(
        subtask_goal=subtask_goal,
        expected_outputs=expected_outputs,
        execution_trace=execution_trace,
        environment_state=environment_state
    )

def get_replanning_prompt(
        original_dag: str,
        failed_subtask: str,
        failure_reason: str,
        current_state: str,
) -> str:
    # get the replanning prompt
    return PromptTemplates.REPLANNING.format(
        original_dag=original_dag,
        failed_subtask=failed_subtask,
        failure_reason=failure_reason,
        current_state=current_state
    )
