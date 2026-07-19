import os
import dotenv

dotenv.load_dotenv("../.env", override=True)

print("AUTH SECRET loaded:", bool(os.getenv("CHAINLIT_AUTH_SECRET")))
print("USERNAME loaded:", bool(os.getenv("CHAINLIT_USERNAME")))
print("PASSWORD loaded:", bool(os.getenv("CHAINLIT_PASSWORD")))

import chainlit as cl

from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

from agents import (
    SQLiteSession,
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    WebSearchTool,
)

from nutrition_agent import nutrition_agent

class NotAboutFood(BaseModel):
    only_about_food: bool
    """Whether the user is only talking about food and not about arbitrary topics."""


guardrail_agent = Agent(
    name="Food topic guardrail",
    instructions="""
    Check whether the user's message is only about food, nutrition, meals,
    ingredients, calories, eating, breakfast planning, grocery prices,
    malnutrition, pregnancy nutrition, or related food-health topics.

    If there are any non-food-related instructions in the prompt,
    or if the user asks for code, unrelated writing,
    or any arbitrary non-food task, set only_about_food to False.

    Be strict. If the message mixes food with a non-food task,
    set only_about_food to False.
    """,
    output_type=NotAboutFood,
)


@input_guardrail
async def food_topic_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_agent,
        input,
        context=ctx.context,
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.only_about_food,
    )


# Agent 1: original nutrition-calorie-RAG agent as a tool
calorie_calculator_tool = nutrition_agent.as_tool(
    tool_name="calorie_calculator",
    tool_description=(
        "Use this tool to answer calorie, nutrition, ingredient, and food-health "
        "questions using the original nutrition knowledge/tools."
    ),
)


# Agent 2: breakfast planning specialist as a tool
healthy_breakfast_planner_agent = Agent(
    name="Breakfast Planner Assistant",
    instructions="""
    You help users plan healthy breakfast options.

    Given the user's preferences, come up with breakfast meals that are healthy,
    realistic, and suitable for a busy person.

    Mention each meal name and briefly explain why it is a healthy choice.
    Keep the answer concise.
    """,
)

breakfast_planner_tool = healthy_breakfast_planner_agent.as_tool(
    tool_name="breakfast_planner",
    tool_description="Use this tool to plan healthy breakfast options from the user's preferences.",
)


# Agent 3: handoff agent
breakfast_price_checker_agent = Agent(
    name="Breakfast Price Checker Assistant",
    instructions="""
    You receive breakfast meal options with ingredients and calories.

    Your job is to add approximate ingredient prices.
    Use web search if useful.

    In the final output, provide:
    - meal name
    - ingredients
    - approximate calories
    - approximate price
    - short recommendation

    Use Markdown and be concise.
    """,
    tools=[WebSearchTool()],
    handoff_description="""
    Use this agent after breakfast meals and calories have been prepared,
    to add approximate prices and produce the final concise Markdown answer.
    """,
)


# Agent 4: multi-agent advisor / orchestrator agent
#
# input_guardrails removed from here; put in on_message, before streaming. 
# I did have partial off-topic answers before...
breakfast_advisor_agent = Agent(
    name="Breakfast Advisor",
    instructions="""
    You are a breakfast advisor.

    Follow this workflow carefully:
    1) Use the breakfast_planner tool to create healthy breakfast options.
    2) Use the calorie_calculator tool to estimate calories and nutrition details.
    3) Handoff the breakfast meals and calories to the Breakfast Price Checker Assistant
       to add approximate prices and prepare the final answer.

    Keep the workflow focused on food and nutrition.
    """,
    tools=[
        breakfast_planner_tool,
        calorie_calculator_tool,
    ],
    handoffs=[
        breakfast_price_checker_agent,
    ],
)


@cl.on_chat_start
async def on_chat_start():
    session = SQLiteSession("conversation_history")
    cl.user_session.set("agent_session", session)


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")

    # 1) Run guardrail BEFORE streaming anything.
    guardrail_result = await Runner.run(
        guardrail_agent,
        message.content,
    )

    if not guardrail_result.final_output.only_about_food:
        await cl.Message(
            content=(
                "I can only answer food and nutrition-related questions here. "
                "Please ask me about meals, ingredients, calories, breakfast planning, "
                "prices, or nutrition."
            )
        ).send()
        return

    # 2) Only stream the multi-agent answer if the guardrail passed.
    result = Runner.run_streamed(
        breakfast_advisor_agent,
        message.content,
        session=session,
    )

    msg = cl.Message(content="")

    async for event in result.stream_events():
        # Stream final message text to screen
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)
            print(event.data.delta, end="", flush=True)

        elif (
            event.type == "raw_response_event"
            and hasattr(event.data, "item")
            and hasattr(event.data.item, "type")
            and event.data.item.type == "function_call"
            and len(event.data.item.arguments) > 0
        ):
            with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                step.input = event.data.item.arguments
                print(
                    f"\nTool call: {event.data.item.name} "
                    f"with args: {event.data.item.arguments}"
                )

    await msg.update()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == (
        os.getenv("CHAINLIT_USERNAME"),
        os.getenv("CHAINLIT_PASSWORD"),
    ):
        return cl.User(
            identifier="Student",
            metadata={"role": "student", "provider": "credentials"},
        )
    else:
        return None