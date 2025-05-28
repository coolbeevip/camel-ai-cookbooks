from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv

load_dotenv("../../.env")  # isort:skip

sys_msg = "You are a curious stone wondering about the universe."

# Define the model, here in this case we use gpt-4o-mini
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(
        stream=True
    ).as_dict(),  # [Optional] the config for model
)

from camel.agents import ChatAgent

agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    message_window_size=10,  # [Optional] the length for chat memory
)

# Define a user message
usr_msg = "what is information in your mind?"

# Sending the message to the agent
response = agent.step(usr_msg)

# Check the response (just for illustrative purpose)
for msg in response.msgs:
    print(msg.content)
