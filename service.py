# demo.py
from modules.chatbot_qwen import QwenChatEngine

# 1️⃣ 建立引擎（只要提供模型路徑即可）
chat_engine = QwenChatEngine(
    model_path="models/Qwen3-8B_int4",
    device="cuda",          # single‑GPU
    dtype="auto",
    quantization="bitsandbytes",   # 4‑bit via bitsandbytes
    trust_remote_code=True,
    max_model_len=7936,    # 若想支援超長上下文
    first_trim_k=5,
)

# 2️⃣ 載入模型與 tokenizer
chat_engine.load()

# 3️⃣ 開始會話（可加入 system prompt）
chat_engine.start_conv(system_prompt="You are a helpful assistant.")

exit_conv = False 
while exit_conv == False:
    usr_input = input("User: ")
    if usr_input == 'exit':
        exit_conv = True
        break
    else:
        resp = chat_engine.chat(
            usr_input,
            enable_thinking=True,
            max_new_tokens=3840,
            temperature=0.55,
            top_p=0.9,
        )
        print("🧠 Thinking:\n", resp["thinking"])
        print("🤖 Answer:\n", resp["answer"])




# 6️⃣ 結束會話、釋放資源
chat_engine.end_conv()
# 若程式結束，__del__ 會自動呼叫 vLLM shutdown

