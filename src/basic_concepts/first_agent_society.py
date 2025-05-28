from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv

load_dotenv("../../.env")  # isort:skip

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(
        temperature=0.0
    ).as_dict(),  # [Optional] the config for model
)

# 设置任务参数
task_kwargs = {
    "task_prompt": "制定一个计划，回到过去，做出改变。",
    "with_task_specify": True,
    "task_specify_agent_kwargs": {"model": model},
}

# 指令发送者
user_role_kwargs = {
    "user_role_name": "一个雄心勃勃的有抱负的时间旅行者",
    "user_agent_kwargs": {"model": model},
}

# 指令执行者
assistant_role_kwargs = {
    "assistant_role_name": "有史以来最好的实验物理学家",
    "assistant_agent_kwargs": {"model": model},
}

# 构造角色扮演社会
society = RolePlaying(
    **task_kwargs,  # The task arguments
    **user_role_kwargs,  # The instruction sender's arguments
    **assistant_role_kwargs,  # The instruction receiver's arguments
)


# 定义终止条件
def is_terminated(response):
    """
    Give alerts when the session should be terminated.
    """
    if response.terminated:
        role = response.msg.role_type.name
        reason = response.info["termination_reasons"]
        print(f"AI {role} terminated due to {reason}")

    return response.terminated


# 定义循环交互
def run(society, round_limit: int = 10):
    # Get the initial message from the ai assistant to the ai user
    input_msg = society.init_chat()

    # Starting the interactive session
    for _ in range(round_limit):
        # Get the both responses for this round
        assistant_response, user_response = society.step(input_msg)

        # Check the termination condition
        if is_terminated(assistant_response) or is_terminated(user_response):
            break

        # Get the results
        print(f"[AI User] {user_response.msg.content}.\n")
        # Check if the task is end
        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
        print(f"[AI Assistant] {assistant_response.msg.content}.\n")

        # Get the input message for the next round
        input_msg = assistant_response.msg

    return None


run(society)
