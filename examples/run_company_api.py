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

from camel.logger import set_log_level
from camel.models import ModelFactory
from camel.toolkits import (
    CompanyDataToolkit
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


def main(company_name):
    toolkit = CompanyDataToolkit()
    # 获取企业信息
    company_info = toolkit.get_company_info(company_name)
    # 获取财务指标
    finance_info = toolkit.get_company_finance_indicator(company_name, "2023-12-31")
    # 搜索微信文章
    articles = toolkit.search_wechat_articles(company_name)
    from collections import namedtuple
    # 定义结构
    Response = namedtuple('Response', 'msgs')
    Message = namedtuple('Message', 'content')
    # 构造数据
    response = Response(msgs=[
        Message(content={
        "company_info": company_info,
        "finance_info": finance_info,
        "articles": articles
        })
    ])
    return response


if __name__ == "__main__":
    default_task = "中国移动"
    task = sys.argv[1] if len(sys.argv) > 1 else default_task
    main(task)
