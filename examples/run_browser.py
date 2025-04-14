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
import sys

from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType

from owl.utils import run_society, DocumentProcessingToolkit
from camel.societies import RolePlaying
from camel.logger import set_log_level

import pathlib

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_society(question: str) -> RolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="deepseek-ai/DeepSeek-V3",
            api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
            url="https://api-inference.modelscope.cn/v1/",
            model_config_dict={"temperature": 0, "max_tokens": 8192},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            # model_type="gpt-4o-mini",
            # api_key=os.getenv("OPENAI_API_KEY"),
            # url="https://free.v36.cm/v1",
            model_type="deepseek-ai/DeepSeek-V3-0324",
            api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
            url="https://api-inference.modelscope.cn/v1/",
            model_config_dict={"temperature": 0, "max_tokens": 2048},
        ),
        "browsing": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="Qwen/Qwen2.5-VL-72B-Instruct",
            api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
            url="https://api-inference.modelscope.cn/v1/",

            # model_type="gemini-2.5-pro-exp-03-25",
            # api_key='AIzaSyDBLczqGDulOjcrOP5SsEGN68fiv7ArVWQ',
            # url="https://generativelanguage.googleapis.com/v1beta/openai/",

            model_config_dict={"temperature": 0.3, "max_tokens": 8192},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="deepseek-ai/DeepSeek-V3",
            api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
            url="https://api-inference.modelscope.cn/v1/",
            model_config_dict={"temperature": 0, "max_tokens": 8192},
        ),
    }

    # Configure toolkits
    tools = [
        *BrowserToolkit(
            headless=False,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
        ).get_tools(),
    ]

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = RolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    return society


def main():
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    # default_task = "通过百度搜索一下, 广赢数字科技公司的主营业务"
    default_task = "利用浏览器工具，访问网站 https://weixin.sogou.com/ ，并在搜索框中输入`Agent`，返回第一个搜索结果的标题和链接"
    # default_task = "浏览亚马逊通过关键字 机械键盘 找到第一个商品。请提供产品名称和价格"
    # Override default task if command line argument is provided
    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    # 构建AI社群
    # 配置各个专门的模型实例
    # 初始化必要的工具集
    society = construct_society(task)

    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()
