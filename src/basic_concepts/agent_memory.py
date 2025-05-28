from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.memories import (
    ChatHistoryBlock,
    LongtermAgentMemory,
    ScoreBasedContextCreator,
    VectorDBBlock,
)
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter
from colorama import Fore
from dotenv import load_dotenv

load_dotenv("../../.env")  # isort:skip

# Initialize the memory
memory = LongtermAgentMemory(
    context_creator=ScoreBasedContextCreator(
        token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
        token_limit=1024,
    ),
    chat_history_block=ChatHistoryBlock(),
    vector_db_block=VectorDBBlock(),
)


# Initialize agent
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig().as_dict(),  # [Optional] the config for model
)
agent = ChatAgent(system_message="You are a helpful assistant.", model=model)

# Set memory to the agent
agent.memory = memory

user_messages = [
    "What is the LLM",
    "What is the capital of China",
    "What is the memory of the agent",
]

for user_msg in user_messages:
    print(Fore.GREEN + "User => " + user_msg)
    response = agent.step(user_msg)
    print(Fore.BLUE + "AI => " + response.msgs[0].content)
    context, token_count = memory.get_context()
    print(Fore.RED + "Memory => " + str(context))
    print(Fore.RED + "Token Count => " + str(token_count))
