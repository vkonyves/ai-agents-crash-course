import chainlit as cl
import dotenv
import os

dotenv.load_dotenv()

from openai.types.responses import ResponseTextDeltaEvent

from pydantic import BaseModel

from agents import (
    SQLiteSession,
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

from nutrition_agent import nutrition_agent


@input_guardrail
async def food_topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.only_about_food),
    )


class NotAboutFood(BaseModel):
    only_about_food: bool
    """Whether the user is only talking about food and not about arbitrary topics"""

guardrail_agent = Agent(
    name="Food topic guardrail",
    instructions="""
    Check whether the user's message is only about food, nutrition, meals,
    ingredients, calories, eating, or related food-health topics.

    If there are any non-food-related instructions in the prompt,
    or if the user asks for code, and unrelated writing, 
    or any arbitrary non-food task, set only_about_food to False.

    Be strict. If the message mixes food with a non-food task,
    set only_about_food to False.
    """,
    output_type=NotAboutFood,
)

guarded_nutrition_agent = nutrition_agent.clone(
    input_guardrails=[food_topic_guardrail],
)


@cl.on_chat_start
async def on_chat_start():
    session = SQLiteSession("conversation_history")
    cl.user_session.set("agent_session", session)


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")

    try:
        result = Runner.run_streamed(
            guarded_nutrition_agent,
            message.content,
            session=session
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

    except InputGuardrailTripwireTriggered:
        await cl.Message(
            content=(
                "I can only answer food and nutrition-related questions here. "
                "Please ask me about meals, ingredients, calories, or nutrition."
            )
        ).send()


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
        return 
    
