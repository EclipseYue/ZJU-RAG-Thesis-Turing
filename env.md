# 激活环境
source /root/snap/ZJU-RAG-Thesis-Turing/venv/bin/activate

# 设置你的 API Key (这里以硅基流动的 Qwen-7B 为例)
export OPENAI_API_KEY="sk-你的真实KEY"
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"

# 跑真实的消融实验 (代码中模型名目前默认是 deepseek-chat，你可以在代码里改成 Qwen/Qwen2.5-7B-Instruct)
python experiments/run_large_scale.py --dataset hotpotqa --samples 500


sk-hquhmcswzugzyihzzqphnjoqfvdlnlswetthnpdkgsiebwzs