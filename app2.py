import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from datetime import datetime

async def main():

    # 設定MCP fetch伺服器參數
    # 這個伺服器用於獲取網頁內容
    fetch_mcp_server = StdioServerParams(command="node", args=["C:/Users/Bob/Documents/Cline/MCP/fetch-mcp/dist/index.js"])

    # 設定MCP文件系統伺服器參數
    # 這個伺服器用於寫入本地文件
    write_mcp_server = StdioServerParams(command="node", args=["C:/Users/Bob/Documents/Cline/MCP/servers/src/filesystem/dist/index.js", "C:/Users/Bob/Desktop/manus_agent"])
    
    # 從MCP伺服器獲取fetch工具
    tools_fetch = await mcp_server_tools(fetch_mcp_server)

    # 從MCP伺服器獲取filesystem工具
    tools_write = await mcp_server_tools(write_mcp_server)

    # 創建內容獲取代理
    # 這個代理負責獲取網頁內容
    fetch_agent = AssistantAgent(
        name="content_fetcher",
        model_client=OllamaChatCompletionClient(model="llama3.1"),
        tools=tools_fetch,
        system_message="你是一個網頁內容獲取助手。使用fetch工具獲取網頁內容。獲取成功後請傳遞給content_rewriter。"
    )
    
    # 創建內容改寫代理
    # 這個代理負責將網頁內容改寫為科技資訊風格
    # 注意：不再在完成時添加TERMINATE，而是將內容傳遞給下一個代理    
    rewriter_agent = AssistantAgent(
        name="content_rewriter",
        model_client=OllamaChatCompletionClient(model="llama3.1"),
        system_message="""你是一個內容改寫專家。將提供給你的網頁內容改寫為科技資訊風格的文章。
        科技資訊風格特點：
        1. 標題簡潔醒目
        2. 開頭直接點明主題
        3. 內容客觀準確但生動有趣
        4. 使用專業術語但解釋清晰
        5. 段落簡短，重點突出
        
        當你完成改寫後，請將內容傳遞給content_writer代理，讓它將你的改寫內容寫入到文件中。"""
    )
    
    # 獲取當前日期並格式化為YYYY-MM-DD
    current_date = datetime.now().strftime('%Y-%m-%d')

    # 創建文件寫入代理
    # 這個代理負責將改寫後的內容寫入本地文件
    # 注意：這個代理會在完成任務後添加TERMINATE來結束對話
    write_agent = AssistantAgent(
        name="content_writer",
        model_client=OllamaChatCompletionClient(model="llama3.1"),
        tools=tools_write,
        system_message=f"""你是一個文件助手。使用filesystem工具將content_rewriter提供的內容寫入txt文件，文件以日期命名（格式為{current_date}.txt）。
        當你成功將文件寫入後，回覆"TERMINATE"以結束對話。"""
    )
    
    # 設置終止條件和團隊
    # 當任何代理回覆TERMINATE時，對話將結束
    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([fetch_agent, rewriter_agent, write_agent], termination_condition=termination)
    
    task = "獲取https://www.aivi.fyi/llms/introduce-Claude-3.7-Sonnet的內容，然後將其改寫為科技資訊風格的文章，然後將改寫的文章寫入本地txt文件"
    
    # 只執行一次任務，使用run方法
    result = await team.run(task=task, cancellation_token=CancellationToken())
    
    # 遍歷並打印所有消息，以顯示整個過程
    print("\n整個對話過程：\n")
    print("-" * 60)
    
    for i, msg in enumerate(result.messages):
        # 判斷消息的類型並相應地打印
        if hasattr(msg, 'source') and hasattr(msg, 'content'):
            print(f"\n---------- {msg.source} ----------")
            print(msg.content)
        elif hasattr(msg, 'source') and hasattr(msg, 'content') and isinstance(msg.content, list):
            print(f"\n---------- {msg.source} (工具調用) ----------")
            for item in msg.content:
                print(item)
        else:
            print(f"\n[消息 {i+1}] (類型: {type(msg).__name__})")
            print(msg)
        
        print("-" * 60)
    
    # 打印最終改寫結果
    print("\n最終改寫結果：\n")
    final_message = result.messages[-1]
    if hasattr(final_message, 'content'):
        print(final_message.content)
    
    return result

# 在Python腳本中運行異步代碼的正確方式
if __name__ == "__main__":
    asyncio.run(main())
