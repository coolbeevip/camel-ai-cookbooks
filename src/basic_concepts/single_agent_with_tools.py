from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import FunctionTool, SearchToolkit
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv

load_dotenv("../../.env")  # isort:skip


# 定义两个工具
def add(a: int, b: int) -> int:
    r"""Adds two numbers.

    Args:
        a (int): The first number to be added.
        b (int): The second number to be added.

    Returns:
        integer: The sum of the two numbers.
    """
    return a + b


def sub(a: int, b: int) -> int:
    r"""Do subtraction between two numbers.

    Args:
        a (int): The minuend in subtraction.
        b (int): The subtrahend in subtraction.

    Returns:
        integer: The result of subtracting :obj:`b` from :obj:`a`.
    """
    return a - b


MATH_FUNCS: list[FunctionTool] = [FunctionTool(func) for func in [add, sub]]

tools_list = [*SearchToolkit().get_tools(), *MATH_FUNCS]

# 定义一个支持工具的模型
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig().as_dict(),  # [Optional] the config for model
)

# Set message for the assistant
assistant_sys_msg = """You are a helpful assistant to do search task."""

agent = ChatAgent(system_message=assistant_sys_msg, model=model, tools=tools_list)

# Set prompt for the search task
prompt_search = """When was University of Oxford set up"""
# Set prompt for the calculation task
prompt_calculate = """Assume now is 2024 in the Gregorian calendar, University of Oxford was set up in 1096, estimate the current age of University of Oxford"""

# Convert the two prompt as message that can be accepted by the Agent
user_msg_search = BaseMessage.make_user_message(role_name="User", content=prompt_search)
user_msg_calculate = BaseMessage.make_user_message(
    role_name="User", content=prompt_calculate
)

# Get response
assistant_response_search = agent.step(user_msg_search)
assistant_response_calculate = agent.step(user_msg_calculate)

print(assistant_response_search.info["tool_calls"])

print(assistant_response_calculate.info["tool_calls"])
