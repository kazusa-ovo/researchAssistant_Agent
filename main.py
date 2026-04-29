import os
import dotenv
from agent import create_research_assistant

dotenv.load_dotenv()

#记忆设置
USER_ID = os.getenv("USER_ID",default="")
SESSION_ID = os.getenv("SESSION_ID",default="")
THREAD_ID = f"{USER_ID}_{SESSION_ID}"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}

#启动Agent
def main():
    print("研究助手 已启动")
    print(f"当前会话: {THREAD_ID}")
    print("你可以问我任何需要研究的问题")
    print("我会先查本地知识库，再搜索互联网")
    print("我会在回答前进行反思，确保答案质量")
    print("\n输入 'esc' 退出对话")

    # 创建 Agent
    agent = create_research_assistant()

    while True:
        user_input = input("\n请输入文本:").strip()
        if user_input == "esc":
            break
        if not user_input:
            continue
        try:
            result = agent.invoke(
                {
                    "messages":[{
                        "role":"user",
                        "content":user_input,
                    }]
                },
                config=CONFIG,
           )

            final_msg = result["messages"][-1]
            if hasattr(final_msg, "content"):
                answer = final_msg.content
            elif isinstance(final_msg, dict):
                answer = final_msg.get("content","")
            else:
                answer = str(final_msg)

            print(f"\n Agent:{answer}")

        except Exception as e:
            print(f"\n Agent运行出错:{e}\n\n请检查网络连接和API配置后重试")

if __name__ == '__main__':
    main()