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
    # 設定 MCP 獲取伺服器參數
    fetch_mcp_server = StdioServerParams(command="node", args=["C:/Users/Bob/Documents/Cline/MCP/fetch-mcp/dist/index.js"])
    
    # 設定MCP文件系統伺服器參數
    # 這個伺服器用於寫入本地文件
    write_mcp_server = StdioServerParams(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/Bob/Desktop/manus_agent"])

    # 從 MCP 伺服器獲取工具
    tools_fetch = await mcp_server_tools(fetch_mcp_server)
    
    # # 從MCP伺服器獲取filesystem工具
    # tools_write = await mcp_server_tools(write_mcp_server)
    
    # 創建獲取代理，並包含 MCP 獲取工具
    fetch_agent = AssistantAgent(
        name="content_fetcher",
        model_client=OllamaChatCompletionClient(model="llama3.1"),
        tools=tools_fetch,  # MCP 獲取工具會在此包含
        system_message="你是一個網頁內容獲取助手。使用fetch工具獲取網頁內容。"
    )
    
    # 創建改寫代理（不變）
    rewriter_agent = AssistantAgent(
        name="content_rewriter",
        model_client=OllamaChatCompletionClient(model="llama3.1"),
        system_message="""你是個內容改寫專家。將提供給你的網頁內容改寫為科技資訊風格的文章。
        科技資訊風格特點：
        1. 標題簡潔醒目
        2. 開頭直接點明主題
        3. 內容客觀準確但生動有趣
        4. 使用專業術語但解釋清晰
        5. 段落簡短，重點突出
        
        當你完成改寫後，回覆TERMINATE。"""
    )
    
    # 設置結束條件（強制執行一次）
    termination = TextMentionTermination("TERMINATE")
    
    # 只執行一次，不進行回合循環
    team = RoundRobinGroupChat([fetch_agent, rewriter_agent], termination_condition=termination)
    
    # 運行工作流程（只執行一次）
    result = await team.run(
        task="獲取https://www.aivi.fyi/llms/introduce-Claude-3.7-Sonnet的內容，然後將其改寫為科技資訊風格的文章",
        cancellation_token=CancellationToken()
    )
    
    print("\n最終改寫結果：\n")
    print(result.messages[-1].content)
    return result

if __name__ == "__main__":
    asyncio.run(main())
