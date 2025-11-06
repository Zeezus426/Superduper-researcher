from typing import Annotated, List, Literal
import operator
from pydantic import BaseModel, Field
from langgraph.types import Send
from typing_extensions import TypedDict
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from lightning import mili, duck
from langchain.messages import ToolMessage
from langchain.messages import AnyMessage


llm = ChatOpenAI(
    model="gpt-5",
    api_key="ahhh",
    base_url="http://127.0.0.1:1234/v1",
)


async def main():
    mili_tools = await mili.get_tools()
    duck_tools = await duck.get_tools()
    
    # Create unified tools_by_name for tool_node (all tools)
    all_tools = mili_tools + duck_tools
    tools_by_name = {tool.name: tool for tool in all_tools}
    
    # Create separate bindings for each tool set
    llm_with_mili = llm.bind_tools(mili_tools)
    llm_with_duck = llm.bind_tools(duck_tools)
    
    
    # Schema for structured output to use in planning
    class Section(BaseModel):
        name: str = Field(
            description="Name for this section of the report.",
        )
        description: str = Field(
            description="Brief overview of the main topics and concepts to be covered in this section.",
        )

    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        llm_calls: int


    class Sections(BaseModel):
        sections: List[Section] = Field(
            description="Sections of the report.",
        )


    # Augment the LLM with schema for structured output
    planner = llm.with_structured_output(Sections)



    # Graph state
    class State(TypedDict):
        topic: str  # Report topic
        sections: list[Section]  # List of report sections
        completed_sections: Annotated[
            list, operator.add
        ]  # All workers write to this key in parallel
        decision: str
        final_report: str  # Final report


    # Worker state
    class WorkerState(TypedDict):
        section: Section
        completed_sections: Annotated[list, operator.add]
        messages: Annotated[list[AnyMessage], operator.add]


    # Nodes
    async def orchestrator(state: State):
        """Orchestrator that generates a plan for the report"""

        # Generate queries
        report_sections = await planner.ainvoke(
            [
                SystemMessage(content="Generate a plan for the report."),
                HumanMessage(content=f"Here is the report topic: {state['topic']}"),
            ]
        )

        return {"sections": report_sections.sections}


    async def llm_call1(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = await llm_with_mili.ainvoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}

    async def llm_call2(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = await llm_with_duck.ainvoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}


    async def llm_call3(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = await llm_with_mili.ainvoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}


    async def llm_call4(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = await llm_with_duck.ainvoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}


    async def llm_call5(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = await llm_with_mili.ainvoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}


    async def llm_call6(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = await llm_with_duck.ainvoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}


    async def llm_call7(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = await llm_with_mili.ainvoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}

    async def synthesizer(state: State):
        """Synthesize full report from sections"""

        # List of completed sections
        completed_sections = state["completed_sections"]

        # Format completed section to str to use as context for final sections
        completed_report_sections = "\n\n---\n\n".join(completed_sections)

        return {"final_report": completed_report_sections}


    # Conditional edge function to create llm_call workers that each write a section of the report
    def assign_workers(state: State):
        """Assign a worker to each section in the plan"""
        for agent in [
            "llm_call1",
            "llm_call2",
            "llm_call3",
            "llm_call4",
            "llm_call5",
            "llm_call6",
            "llm_call7"
        ]:
        # Kick off section writing in parallel via Send() API
            return [Send(agent, {"section": s}) for s in state["sections"]]
        

    async def tool_node(state: dict):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.ainvoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    

        
    def route_after_tools(state: State):
        """Route back to the appropriate agent after tool execution"""
        decision = state.get("decision")
        mapping = {
            "llm_call1":"llm_call1",
            "llm_call2":"llm_call2",
            "llm_call3":"llm_call3",
            "llm_call4":"llm_call4",
            "llm_call5":"llm_call5",
            "llm_call6":"llm_call6",
            "llm_call7":"llm_call7",

        }
        return mapping, decision


    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END

    # Build workflow
    orchestrator_worker_builder = StateGraph(State)

    # Add the nodes
    orchestrator_worker_builder.add_node("orchestrator", orchestrator)
    orchestrator_worker_builder.add_node("llm_call1", llm_call1)
    orchestrator_worker_builder.add_node("llm_call2", llm_call2)
    orchestrator_worker_builder.add_node("llm_call3", llm_call3)
    orchestrator_worker_builder.add_node("llm_call4", llm_call4)
    orchestrator_worker_builder.add_node("llm_call5", llm_call5)
    orchestrator_worker_builder.add_node("llm_call6", llm_call6)
    orchestrator_worker_builder.add_node("llm_call7", llm_call7)
    orchestrator_worker_builder.add_node("tool_node", tool_node)

    orchestrator_worker_builder.add_node("synthesizer", synthesizer)


        # 1. START → orchestrator (acts as llm_call_router)
    orchestrator_worker_builder.add_edge(START, "orchestrator")

    # 2. orchestrator → conditionally to any agent (llm_call1 to llm_call7)
    orchestrator_worker_builder.add_conditional_edges(
        "orchestrator",
        assign_workers,  # same as alphagraph's route_decision
        {agent: agent for agent in [
            "llm_call1", "llm_call2", "llm_call3", "llm_call4", 
            "llm_call5", "llm_call6", "llm_call7"
        ]}
    )

    # 3. Each agent → either tool_node or END (exactly like alphagraph)
    for agent in ["llm_call1", "llm_call2", "llm_call3", "llm_call4", 
                "llm_call5", "llm_call6", "llm_call7"]:
        orchestrator_worker_builder.add_conditional_edges(
            agent,
            should_continue,  # same as alphagraph
            {
                "tool_node": "tool_node",  # continue looping
                END: END,                  # finish, trigger synthesizer
            }
        )

    # 4. After tool execution → go back to the SAME agent (alphagraph loop)
    orchestrator_worker_builder.add_conditional_edges(
        "tool_node",
        route_after_tools,  # same as alphagraph
        {agent: agent for agent in [
            "llm_call1", "llm_call2", "llm_call3", "llm_call4", 
            "llm_call5", "llm_call6", "llm_call7"
        ]}
    )

    # 5. ✅ CRITICAL: When ANY agent reaches END, go to synthesizer
    #    This mimics: “after final decision, synthesize output”
    orchestrator_worker_builder.add_edge("synthesizer", END)

    orchestrator_worker = orchestrator_worker_builder.compile()


    # Invoke
    state = await orchestrator_worker.ainvoke({"topic": "Using the tools available to you research and Create a report on LLM scaling laws"})

    print(state["final_report"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())