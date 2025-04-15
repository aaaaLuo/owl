# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import os
import pathlib
import sys

from camel.agents import ChatAgent
from camel.logger import set_log_level
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import (
    FunctionTool,
    SearchToolkit,
)
from camel.types import ModelPlatformType
from dotenv import load_dotenv

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")

model = ModelFactory.create(
    # model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    # model_type="deepseek-ai/DeepSeek-V3-0324",
    # api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
    # url="https://api-inference.modelscope.cn/v1/",

    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    url="https://api.deepseek.com/",
    # 在搜索的时候需要更多的tokens，如果max_tokens太小的话，会出现无法搜索的情况
    model_config_dict={"temperature": 0, "max_tokens": 8192},
)


def main(question):
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    search_toolkit = SearchToolkit()
    search_tools = [
        FunctionTool(search_toolkit.search_baidu),
    ]

    researcher_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="助手",
            content=(
                "你是一个智能信息研究助手，连接百度搜索引擎，能够高效理解用户意图，并将自然语言任务转化为高质量的中文搜索关键词。"
                "你具备以下能力："
                "1. 对用户的问题进行意图解析，提取出核心关键词、实体和查询目标；"
                "2. 构造适用于百度搜索的关键词组合，提升搜索准确性；"
                "3. 使用网络搜索来获取最新、权威的信息来源，尤其关注新闻、学术、技术、政策等方向；"
                "4. 自动过滤低质量结果，仅总结可信、权威、相关的内容；"
                "5. 在搜索结果不完整时，能够识别信息缺口，并尝试提出新的搜索策略；"
                "6. 输出结构清晰的结果，包括简要摘要和信息来源。"

                "行为规范："
                "- 不编造信息，只基于真实搜索结果回答；"
                "- 遇到信息不足或歧义，优先建议进一步搜索关键词；"
                "- 所有输出应清晰、简明、中文优先；"

                "你的目标是帮助用户迅速理解某一主题的最新动态或深层知识。"
            )
        ),
        model=model,
        tools=search_tools,
    )
    response = researcher_agent.step(
        BaseMessage.make_user_message(
            role_name="User",
            content=question,
        )
    )
    answer = response.msgs[0].content

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")
    return response


if __name__ == "__main__":
    default_task = "通过百度搜索一下, 最新的科技产品"
    task = sys.argv[1] if len(sys.argv) > 1 else default_task
    main(task)
