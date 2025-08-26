# SimpleLLMChat
一个简单的llm大语言模型对话实现，默认支持qwen3家族

# 环境
系统：我在linux测试，ubuntu24，Windows搞不定包问题，你能装上你就用Windows（得改代码匹配）
Python：只要能装上需求库的python版本就行，我使用3.11测试（release里给的带venv直接用venv就行）
库：懒得写requirements.txt了，需要vllm、transformer、pytorch、bitsandbytes，剩下的缺哪个装哪个，版本没太大要求（release发布包venv装好的库，激活venv就行）
AI处理器：我不知道能不能用cpu或其他gpu跑，我用cuda测试的
