# LLM API 配置说明

本文档只说明如何配置，不代表会自动触发真实实验。

## 1. OpenAI 兼容接口

项目中的生成与验证后端统一走 OpenAI 兼容调用方式，因此可以接入：

- OpenAI 官方接口
- Moonshot / Kimi（历史可选）
- SiliconFlow（历史可选）
- 其他支持 `chat.completions` 的兼容网关

## 2. 环境变量

### 2.1 OpenAI

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 2.2 DeepSeek

```bash
export DEEPSEEK_API_KEY="sk-..."
# 可选
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

### 2.3 Moonshot / Kimi（历史可选）

```bash
export MOONSHOT_API_KEY="sk-..."
# 可选
export MOONSHOT_BASE_URL="https://api.moonshot.cn/v1"
```

也兼容：

```bash
export KIMI_API_KEY="sk-..."
```

### 2.4 SiliconFlow（历史可选）

```bash
export SILICONFLOW_API_KEY="sk-..."
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"
```

## 3. 命令行参数

主实验入口支持：

```bash
--generator-backend auto|heuristic|openai|deepseek|moonshot|siliconflow
--generator-model <model_name>
--verifier-backend heuristic|openai|deepseek|moonshot|siliconflow
--verifier-model <model_name>
--generator-api-key <api_key>
--generator-base-url <base_url>
--verifier-api-key <api_key>
--verifier-base-url <base_url>
--real-cove
```

其中：

- `--real-cove` 用于显式切换到真实 LLM 验证。
- 不加该参数时，默认保留现有“类 CoVe / 启发式近似验证”路线。
- 推荐把敏感 key 放进本地覆盖文件，而不是直接写在命令行。

## 4. 本地私有覆盖文件

推荐在本地创建：

```text
experiments/configs/local_api_overrides.json
```

该文件会被主实验脚本和主要诊断脚本自动读取，适合存放：

- `generator_backend`
- `generator_model`
- `generator_api_key`
- `generator_base_url`
- `verifier_backend`
- `verifier_model`
- `verifier_api_key`
- `verifier_base_url`
- `real_cove`

## 5. 推荐原则

- 只想稳定复现实验：使用默认启发式验证
- 当前默认真实 API：DeepSeek
- 想验证真实 LLM 生成：设置 `generator-backend`
- 想补真实 CoVe/NLI 风格验证：设置 `verifier-backend` 并加 `--real-cove`

## 6. 注意

- 当前论文中的主结果仍以启发式验证与启发式生成为主，避免和真实 API 跑数混淆。
- 如果后续要正式改用真实后端，应重新生成结果 JSON、图表和论文结论。
