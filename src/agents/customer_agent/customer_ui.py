import json
import os
import traceback
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from py_trees.blackboard import Blackboard

from src.agents.customer_agent.customer_agent import CustomerServiceSystem, \
    CustomerContext

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    st.set_page_config(page_title="智能客服系统", page_icon="🤖", layout="wide")

    st.title("🤖 智能客服系统")
    st.info("行为树 × 生成式AI 驱动的智能客户服务")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # OpenAI API 配置
        api_url = st.text_input(
            "OPENAI_BASE_URL",
            type="password",
            help="请输入您的OpenAI API地址",
            value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        api_key = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            help="请输入您的OpenAI API密钥",
            value=os.environ.get("OPENAI_API_KEY", ""),
        )

        if not api_url:
            st.warning("请先配置OpenAI API Url")
            st.stop()

        if not api_key:
            st.warning("请先配置OpenAI API Key")
            st.stop()

        # 客户信息
        st.header("👤 客户信息")
        user_id = st.text_input("客户ID", value="CUST_001")

        # 系统状态
        st.header("📊 系统状态")
        if "customer_context" in st.session_state:
            context = st.session_state.customer_context
            st.metric("对话轮次", len(context.conversation_history) // 2)
            st.metric("当前优先级", context.priority.name)
            if context.sentiment:
                st.metric("客户情感", context.sentiment)

    # 初始化系统
    if "cs_system" not in st.session_state:
        try:
            st.session_state.cs_system = CustomerServiceSystem(api_key)
            st.success("✅ 智能客服系统已初始化")
        except Exception as e:
            st.error(f"❌ 系统初始化失败: {e}")
            st.stop()

    # 初始化客户上下文
    if "customer_context" not in st.session_state:
        st.session_state.customer_context = CustomerContext(
            user_id=user_id, conversation_history=[]
        )

    # 主要界面布局
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 对话界面")

        # 显示对话历史
        chat_container = st.container()
        with chat_container:
            if st.session_state.customer_context.conversation_history:
                for i, msg in enumerate(
                    st.session_state.customer_context.conversation_history
                ):
                    if "user" in msg:
                        st.chat_message("user").write(msg["user"])
                    elif "assistant" in msg:
                        st.chat_message("assistant").write(msg["assistant"])

        # 输入界面
        user_input = st.chat_input("请输入您的问题...")

        if user_input:
            # 显示用户消息
            st.chat_message("user").write(user_input)

            # 处理消息
            with st.spinner("正在处理您的问题..."):
                try:
                    response_data = st.session_state.cs_system.process_message(
                        user_input, st.session_state.customer_context
                    )

                    # 显示系统回复
                    st.chat_message("assistant").write(response_data.message)

                    # 显示建议操作
                    if response_data.suggested_actions:
                        with st.expander("💡 建议操作"):
                            for action in response_data.suggested_actions:
                                st.write(f"• {action}")

                    # 重新运行以更新界面
                    st.rerun()

                except Exception as e:
                    st.error(f"处理消息时出错: {e}")
                    st.expander("查看错误详情").code(traceback.format_exc())

    with col2:
        st.header("🔍 分析面板")

        # 行为树状态
        st.subheader("🌳 行为树状态")
        if hasattr(st.session_state, "cs_system"):
            tree_status = (
                "运行中" if st.session_state.cs_system.behavior_tree else "未初始化"
            )
            st.write(f"状态: {tree_status}")

            # 显示行为树结构
            st.image(
                os.path.join(current_dir, "customer_agent.png"),
                caption="行为树结构",
                use_container_width=True,
            )

        # 最新分析结果
        st.subheader("📈 最新分析")
        if st.session_state.customer_context.current_intent:
            st.write(f"**意图**: {st.session_state.customer_context.current_intent}")
        if st.session_state.customer_context.sentiment:
            st.write(f"**情感**: {st.session_state.customer_context.sentiment}")
        st.write(f"**优先级**: {st.session_state.customer_context.priority.name}")

        # 控制按钮
        st.subheader("🎛️ 控制面板")
        if st.button("🔄 重置对话"):
            st.session_state.customer_context = CustomerContext(
                user_id=user_id, conversation_history=[]
            )
            Blackboard.clear()  # 清除行为树黑板数据
            st.success("对话已重置")
            st.rerun()

        if st.button("📊 导出对话"):
            conversation_data = {
                "user_id": st.session_state.customer_context.user_id,
                "conversation_history": st.session_state.customer_context.conversation_history,
                "summary": {
                    "total_messages": len(
                        st.session_state.customer_context.conversation_history
                    ),
                    "final_intent": st.session_state.customer_context.current_intent,
                    "final_sentiment": st.session_state.customer_context.sentiment,
                    "priority": st.session_state.customer_context.priority.name,
                },
            }
            st.download_button(
                label="下载对话记录",
                data=json.dumps(conversation_data, ensure_ascii=False, indent=2),
                file_name=f"conversation_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()