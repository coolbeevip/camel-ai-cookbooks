from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv("../../.env")  # isort:skip

sys_msg = """你是一个数据抓取助手，可以根据用户的要求抓取数据。注意不要过程直接输出 JSON 格式的数据。

# 任务描述
根据用户提供的要求，抓取并返回以下数据：
- MSISDN: 移动台国际用户目录号码或者故障号码，通常为 10 到 15 位数字
- IMSI: 移动用户身份码，通常为 15 位数字，但可以更短，最常见的是 14 位
- APN: 接入点名称
- ServiceCode

# 输出格式
输出应为 JSON 格式，包含上述四个字段。每个字段的值应根据实际抓取到的数据填写。

# 示例

## 示例 1
**输入:**
故障号码1441028323964（IMSI：460082283203964）签约APN为CMIOTZSZCGRE.GD，用户卡（MSISDN:1441028323964，IMSI:460082283203964）

**输出:**
```json
{
  "MSISDN": "1441028323964",
  "IMSI": "460082283203964",
  "APN": "CMIOTZSZCGRE.GD",
  "Service Code": null
}
```

# 注意事项
- 确保输出的 JSON 格式正确无误。
- 如果某个字段没有抓取到数据，可以将其值设为 `null`。
- 抓取数据时，确保数据的准确性和完整性。"""


class ResponseFormat(BaseModel):
    msisdn: str
    imsi: str
    apn: str
    service_code: str


# model = ModelFactory.create(
#     model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
#     model_type="qwen2.5-72b-instruct",
#     model_config_dict=ChatGPTConfig(temperature=0.8, n=1, max_tokens=2000).as_dict(), # response_format=ResponseFormat
#     url=os.environ.get("DASHSCOPE_API_BASE_URL"),
#     api_key=os.environ.get("DASHSCOPE_API_KEY"),
# )

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(
        temperature=0.6, response_format=ResponseFormat
    ).as_dict(),
)


agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    message_window_size=10,  # [Optional] the length for chat memory
)

# Define a user message
# usr_msg = "what is information in your mind?"
usr_msg = input("请输入查询内容 (输入 'quit' 退出): ")
# Sending the message to the agent
response = agent.step(usr_msg)

print("\n")
print(response.msgs[0].content)
