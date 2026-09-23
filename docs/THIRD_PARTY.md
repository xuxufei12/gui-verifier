# 第三方模型与数据

核对日期：2026-09-23。项目代码 MIT 不覆盖以下资产。

| 资产 | 上游声明 | 来源与用途 |
|---|---|---|
| Qwen3.5-0.8B | Apache-2.0 | https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE ，基础模型，用户自行下载 |
| OS-Atlas-data | 数据卡 Apache-2.0 | https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data ，截图来源 |
| ScreenQA | CC BY 4.0 | https://github.com/google-research-datasets/screen_qa/blob/main/LICENSE ，截图/问答来源 |
| ScreenSpot-v2 | 本地上游记录 Apache-2.0；发布任何派生资产前再核对 | https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2 ，历史派生评测和重叠排除参考 |

许可证标签不自动解决截图中的第三方 UI、商标、文本及个人信息权利。公开包包含派生标注JSONL（新增标注CC BY 4.0）和来源映射，不包含原始截图、原始教师响应或模型权重。将来发布适配器和派生标签时，需要保留基础模型许可、来源、版本和所需致谢，检查数据及教师服务对派生资产的条件。

依赖代码请按各自发行版 LICENSE 处理。本包使用 Transformers、PyTorch、PEFT、FLA 等库，通过包管理安装，不重新打包依赖源代码。
