import os

from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv
from camel.agents import ChatAgent

load_dotenv("../../.env")  # isort:skip

sys_msg = "你是一块好奇的石头，思考着宇宙。"

# Define the model, here in this case we use qwen1.5-72b-chat
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="qwen1.5-72b-chat",
    model_config_dict=ChatGPTConfig(temperature=0.8, n=1, max_tokens=2000).as_dict(),
    url=os.environ.get("DASHSCOPE_API_BASE_URL"),
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
)

agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    message_window_size=10,  # [Optional] the length for chat memory
)

# Define a user message
usr_msg = "你脑中的信息是什么？"

# Sending the message to the agent
response = agent.step(usr_msg)

# Check the response (just for illustrative purpose)
print(response.msgs[0].content)
