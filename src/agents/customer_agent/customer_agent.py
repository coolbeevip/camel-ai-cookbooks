import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import py_trees

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.base.py_trees_util import safe_get_blackboard, safe_set_blackboard

load_dotenv("../../../.env")  # isort:skip

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))


# 数据模型
class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class ConversationStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    PENDING = "pending"


class Intent(BaseModel):
    intent: str = Field(..., min_length=1, description="具体意图类别，不能为空")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度，范围在0.0到1.0")
    sentiment: str = Field(
        ...,
        pattern="^(positive|neutral|negative)$",
        description="情感状态，只能是 positive, neutral, 或 negative",
    )
    priority: str = Field(
        ...,
        pattern="^(low|medium|high|urgent)$",
        description="优先级，只能是 low, medium, high, urgent",
    )
    entities: List[str] = Field(default_factory=list, description="提取的关键实体")
    context_needed: bool = Field(default=False, description="是否需要上下文")


class ResponseModel(BaseModel):
    message: str = Field(..., min_length=1, description="客服回复内容，不能为空")
    suggested_actions: List[str] = Field(
        ..., description="建议的后续行动，非空字符串数组"
    )
    escalation_needed: bool = Field(..., description="是否需要升级")
    follow_up_questions: List[str] = Field(
        ..., description="可能的跟进问题，非空字符串数组"
    )


@dataclass
class CustomerContext:
    """客户上下文信息"""

    user_id: str
    conversation_history: List[Dict[str, str]]
    current_intent: Optional[str] = None
    sentiment: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    satisfaction_score: Optional[float] = None
    resolved_issues: Optional[List[str]] = None

    def __post_init__(self):
        if self.resolved_issues is None:
            self.resolved_issues = []


@dataclass
class ResponseData:
    """响应数据"""

    message: str
    intent: str
    confidence: float
    suggested_actions: List[str]
    escalation_needed: bool = False


# AI服务类
class AIService:
    """集成OpenAI的AI服务"""

    def __init__(self, api_key: str, url: Optional[str] = None):
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict=ChatGPTConfig().as_dict(),
            api_key=api_key,
            url=url,
        )
        self.agent = ChatAgent(model=model, message_window_size=10)

    def analyze_intent(self, message: str, context: CustomerContext) -> Dict[str, Any]:
        """分析用户意图"""
        prompt = f"""
        分析以下客户消息的意图，并返回JSON格式结果：

        客户消息: "{message}"
        历史对话: {json.dumps(context.conversation_history[-3:], ensure_ascii=False)}

        请返回以下格式的JSON：
        {{
            "intent": "具体意图类别",
            "confidence": 0.0-1.0的置信度,
            "sentiment": "positive/neutral/negative",
            "priority": "low/medium/high/urgent",
            "entities": ["提取的关键实体"],
            "context_needed": true/false
        }}

        意图类别包括：问候、产品咨询、技术支持、投诉、退款、账户问题、其他
        """

        try:
            self.agent.reset()
            response = self.agent.step(prompt, response_format=Intent)
            result = json.loads(response.msgs[0].content)
            return result
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "intent": "其他",
                "confidence": 0.5,
                "sentiment": "neutral",
                "priority": "medium",
                "entities": [],
                "context_needed": False,
            }

    def generate_response(
        self, message: str, context: CustomerContext, intent_data: Dict
    ) -> ResponseData:
        """生成客服回复"""
        conversation_history = "\n".join(
            [
                f"客户: {msg['user']}" if "user" in msg else f"客服: {msg['assistant']}"
                for msg in context.conversation_history[-5:]
            ]
        )

        prompt = f"""
        你是一个专业的客服代表。根据以下信息生成合适的回复：

        客户最新消息: "{message}"
        意图分析: {json.dumps(intent_data, ensure_ascii=False)}
        对话历史: {conversation_history}
        客户情感: {context.sentiment}
        优先级: {context.priority.name}

        请生成一个JSON格式的回复：
        {{
            "message": "客服回复内容",
            "suggested_actions": ["建议的后续行动"],
            "escalation_needed": true/false,
            "follow_up_questions": ["可能的跟进问题"]
        }}

        回复要求：
        1. 专业、友好、有帮助
        2. 根据客户情感调整语调
        3. 提供具体的解决方案
        4. 如果无法解决，建议升级处理
        """

        try:
            self.agent.reset()
            response = self.agent.step(prompt, response_format=ResponseModel)
            result = json.loads(response.msgs[0].content)
            return ResponseData(
                message=result.get("message", "我理解您的问题，让我为您查找解决方案。"),
                intent=intent_data["intent"],
                confidence=intent_data["confidence"],
                suggested_actions=result.get("suggested_actions", []),
                escalation_needed=result.get("escalation_needed", False),
            )
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return ResponseData(
                message="抱歉，我遇到了一些技术问题。请稍等片刻，或者我可以为您转接人工客服。",
                intent=intent_data["intent"],
                confidence=0.5,
                suggested_actions=["转接人工客服"],
                escalation_needed=True,
            )


# 行为树节点定义
class AnalyzeIntentNode(py_trees.behaviour.Behaviour):
    """意图分析节点"""

    def __init__(self, name: str, ai_service: AIService):
        super().__init__(name)
        self.ai_service = ai_service

    def update(self) -> py_trees.common.Status:
        try:
            blackboard = self.attach_blackboard_client(name="CustomerService")
            blackboard.register_key(
                key="customer_context", access=py_trees.common.Access.READ
            )
            blackboard.register_key(
                key="current_message", access=py_trees.common.Access.READ
            )

            message = safe_get_blackboard(blackboard, "current_message")
            context = safe_get_blackboard(blackboard, "customer_context")

            if not message or not context:
                return py_trees.common.Status.FAILURE

            # 执行意图分析
            intent_data = self.ai_service.analyze_intent(message, context)

            # 更新上下文
            context.current_intent = intent_data["intent"]
            context.sentiment = intent_data["sentiment"]
            context.priority = Priority[intent_data["priority"].upper()]

            # 保存结果到黑板
            blackboard.register_key(
                key="intent_data", access=py_trees.common.Access.WRITE
            )
            blackboard.register_key(
                key="customer_context", access=py_trees.common.Access.WRITE
            )
            safe_set_blackboard(blackboard, "intent_data", intent_data)
            safe_set_blackboard(blackboard, "customer_context", context)

            logger.info(
                f"Intent analyzed: {intent_data['intent']} (confidence: {intent_data['confidence']})"
            )
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            logger.error(f"Intent analysis node failed: {e}")
            return py_trees.common.Status.FAILURE


class CheckEscalationNode(py_trees.behaviour.Behaviour):
    """检查是否需要升级节点"""

    def __init__(self, name: str):
        super().__init__(name)

    def update(self) -> py_trees.common.Status:
        try:
            blackboard = self.attach_blackboard_client(name="CustomerService")
            blackboard.register_key(
                key="customer_context", access=py_trees.common.Access.READ
            )
            blackboard.register_key(
                key="intent_data", access=py_trees.common.Access.READ
            )

            if not blackboard.exists("customer_context") or not blackboard.exists(
                "intent_data"
            ):
                return py_trees.common.Status.FAILURE

            context = safe_get_blackboard(blackboard, "customer_context")
            intent_data = safe_get_blackboard(blackboard, "intent_data")

            # 升级条件检查
            escalation_needed = (
                context.priority == Priority.URGENT
                or intent_data["sentiment"] == "negative"
                and intent_data["confidence"] > 0.8
                or "投诉" in intent_data["intent"]
                or "退款" in intent_data["intent"]
                or len(context.conversation_history) > 10  # 对话过长
            )

            blackboard.register_key(
                key="escalation_needed", access=py_trees.common.Access.WRITE
            )
            safe_set_blackboard(blackboard, "escalation_needed", escalation_needed)

            if escalation_needed:
                logger.info("Escalation needed")
                return py_trees.common.Status.SUCCESS
            else:
                logger.info("No escalation needed")
                return py_trees.common.Status.FAILURE

        except Exception as e:
            logger.error(f"Escalation check failed: {e}")
            return py_trees.common.Status.FAILURE


class GenerateResponseNode(py_trees.behaviour.Behaviour):
    """生成回复节点"""

    def __init__(self, name: str, ai_service: AIService):
        super().__init__(name)
        self.ai_service = ai_service

    def update(self) -> py_trees.common.Status:
        try:
            blackboard = py_trees.blackboard.Client(name="CustomerService")
            blackboard.register_key(
                key="current_message", access=py_trees.common.Access.READ
            )
            blackboard.register_key(
                key="customer_context", access=py_trees.common.Access.READ
            )
            blackboard.register_key(
                key="intent_data", access=py_trees.common.Access.READ
            )

            message = safe_get_blackboard(blackboard, "current_message")
            context = safe_get_blackboard(blackboard, "customer_context")
            intent_data = safe_get_blackboard(blackboard, "intent_data")

            if not all([message, context, intent_data]):
                return py_trees.common.Status.FAILURE

            # 生成回复
            response_data = self.ai_service.generate_response(
                message, context, intent_data
            )

            # 更新对话历史
            context.conversation_history.append({"user": message})
            context.conversation_history.append({"assistant": response_data.message})

            # 保存结果
            blackboard.register_key(
                key="response_data", access=py_trees.common.Access.WRITE
            )
            blackboard.register_key(
                key="customer_context", access=py_trees.common.Access.WRITE
            )
            safe_set_blackboard(blackboard, "response_data", response_data)
            safe_set_blackboard(blackboard, "customer_context", context)

            logger.info(f"Response generated: {response_data.message[:50]}...")
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return py_trees.common.Status.FAILURE


class EscalateToHumanNode(py_trees.behaviour.Behaviour):
    """升级到人工客服节点"""

    def __init__(self, name: str):
        super().__init__(name)

    def update(self) -> py_trees.common.Status:
        try:
            blackboard = py_trees.blackboard.Client(name="CustomerService")
            blackboard.register_key(
                key="customer_context", access=py_trees.common.Access.READ
            )

            context = safe_get_blackboard(blackboard, "customer_context")
            if not context:
                return py_trees.common.Status.FAILURE

            escalation_message = """
            我理解您的问题比较复杂，为了给您提供更好的服务，
            我现在为您转接专业的人工客服。请稍等片刻，
            人工客服将很快为您提供帮助。

            转接编号：CS-{timestamp}
            预计等待时间：3-5分钟
            """.format(timestamp=int(time.time()))

            response_data = ResponseData(
                message=escalation_message,
                intent="escalation",
                confidence=1.0,
                suggested_actions=["等待人工客服", "留下联系方式"],
                escalation_needed=True,
            )

            # 更新对话历史
            context.conversation_history.append({"assistant": escalation_message})

            blackboard.register_key(
                key="response_data", access=py_trees.common.Access.WRITE
            )
            blackboard.register_key(
                key="customer_context", access=py_trees.common.Access.WRITE
            )
            safe_set_blackboard(blackboard, "response_data", response_data)
            safe_set_blackboard(blackboard, "customer_context", context)

            logger.info("Escalated to human agent")
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            return py_trees.common.Status.FAILURE


class LogInteractionNode(py_trees.behaviour.Behaviour):
    """记录交互日志节点"""

    def __init__(self, name: str):
        super().__init__(name)

    def update(self) -> py_trees.common.Status:
        try:
            blackboard = py_trees.blackboard.Client(name="CustomerService")
            blackboard.register_key(
                key="customer_context", access=py_trees.common.Access.READ
            )
            blackboard.register_key(
                key="intent_data", access=py_trees.common.Access.READ
            )
            blackboard.register_key(
                key="response_data", access=py_trees.common.Access.READ
            )

            context = safe_get_blackboard(blackboard, "customer_context")
            intent_data = safe_get_blackboard(blackboard, "intent_data")
            response_data = safe_get_blackboard(blackboard, "response_data")

            if not all([context, intent_data, response_data]):
                return py_trees.common.Status.FAILURE

            # 记录交互日志
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": context.user_id,
                "intent": intent_data["intent"],
                "sentiment": intent_data["sentiment"],
                "priority": context.priority.name,
                "escalated": response_data.escalation_needed,
                "confidence": intent_data["confidence"],
            }

            # 这里可以保存到数据库
            logger.info(f"Interaction logged: {json.dumps(log_entry)}")

            return py_trees.common.Status.SUCCESS

        except Exception as e:
            logger.error(f"Logging failed: {e}")
            return py_trees.common.Status.FAILURE


# 客服系统主类
class CustomerServiceSystem:
    """智能客服系统主类"""

    def __init__(self, openai_api_key: str, openai_api_url: Optional[str] = None):
        self.ai_service = AIService(openai_api_key, openai_api_url)
        self.behavior_tree = self._build_behavior_tree()
        py_trees.display.render_dot_tree(
            self.behavior_tree.root, target_directory=current_dir, name="customer_agent"
        )

    def _build_behavior_tree(self) -> py_trees.trees.BehaviourTree:
        """构建行为树"""

        # 根节点 - 选择器
        root = py_trees.composites.Selector(name="customer_agent", memory=False)

        # 升级分支 - 序列
        escalation_branch = py_trees.composites.Sequence(
            name="EscalationBranch", memory=False
        )
        escalation_branch.add_children(
            [
                CheckEscalationNode("CheckEscalation"),
                EscalateToHumanNode("EscalateToHuman"),
                LogInteractionNode("LogEscalation"),
            ]
        )

        # 常规处理分支 - 序列
        normal_branch = py_trees.composites.Sequence(name="NormalBranch", memory=False)
        normal_branch.add_children(
            [
                AnalyzeIntentNode("AnalyzeIntent", self.ai_service),
                GenerateResponseNode("GenerateResponse", self.ai_service),
                LogInteractionNode("LogInteraction"),
            ]
        )

        # 添加分支到根节点
        root.add_children([escalation_branch, normal_branch])

        return py_trees.trees.BehaviourTree(root)

    def process_message(self, message: str, context: CustomerContext) -> ResponseData:
        """处理客户消息"""

        # 创建并注册黑板客户端，指定写入权限
        blackboard = py_trees.blackboard.Client(name="CustomerService")
        blackboard.register_key(
            key="current_message", access=py_trees.common.Access.WRITE
        )
        blackboard.register_key(
            key="customer_context", access=py_trees.common.Access.WRITE
        )

        # 设置黑板数据
        safe_set_blackboard(blackboard, "current_message", message)
        safe_set_blackboard(blackboard, "customer_context", context)

        # 执行行为树
        self.behavior_tree.tick()

        # 获取结果
        blackboard.register_key(key="response_data", access=py_trees.common.Access.READ)
        blackboard.register_key(
            key="customer_context", access=py_trees.common.Access.READ
        )
        response_data = safe_get_blackboard(blackboard, "response_data")
        updated_context = safe_get_blackboard(blackboard, "customer_context")

        if response_data and updated_context:
            # 更新传入的上下文对象
            context.conversation_history = updated_context.conversation_history
            context.current_intent = updated_context.current_intent
            context.sentiment = updated_context.sentiment
            context.priority = updated_context.priority

            return response_data
        else:
            # 默认回复
            return ResponseData(
                message="抱歉，我暂时无法处理您的请求。请稍后再试或联系人工客服。",
                intent="unknown",
                confidence=0.0,
                suggested_actions=["重试", "联系人工客服"],
                escalation_needed=True,
            )



