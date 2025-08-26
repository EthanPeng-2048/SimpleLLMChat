# qwen_chat_engine.py
from __future__ import annotations

from typing import List, Dict, Optional, Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import re
_SURROGATE_RE = re.compile(r'[\uD800-\uDFFF]')

class QwenChatEngine:
    """
    薄封装：vLLM + HuggingFace tokenizer → 简洁会话 API
    具备：
    - 思考（</think>）不写入历史
    - 自动 token‑budget 管理（窗口裁剪 + 摘要压缩）
    """

    # --------------------------------------------------------------------- #
    # 初始化與載入相關
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "auto",
        quantization: str = "bitsandbytes",
        trust_remote_code: bool = True,
        tokenizer_path: Optional[str] = None,
        max_model_len: Optional[int] = None,
        # 新增默认保留轮次
        first_trim_k: int = 6,
    ) -> None:
        """
        Parameters
        ----------
        model_path : str
            本地模型目錄（必須包含 safetensors 權重）。
        device : str, default "cuda"
        dtype : str, default "auto"
        quantization : str, default "bitsandbytes"
        trust_remote_code : bool, default True
        tokenizer_path : Optional[str]
        max_model_len : Optional[int]
            若為 ``None``，內部預設 8192（保守值）。
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.tokenizer_path = tokenizer_path or model_path
        self.max_model_len = max_model_len
        self.first_trim_k = first_trim_k

        # 內部狀態
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.conversation: List[Dict[str, str]] = []
        self._think_end_token_id: Optional[int] = None

        # 預設 sampling 參數
        self.default_sampling_kwargs: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 512,
        }
    def set_first_trim_k(self, k: int) -> None:
        """运行时修改第一次裁切保留的轮次数。"""
        if k < 0:
            raise ValueError("k 必须是非负整数")
        self.first_trim_k = k

    def _clean_text(text: str) -> str:
        return _SURROGATE_RE.sub('�', text)   # 替换为 �，或直接 '' 删除

    # --------------------------------------------------------------------- #
    # 核心載入
    # --------------------------------------------------------------------- #
    def load(self) -> None:
        """載入 tokenizer 與 vLLM LLM 實例。"""
        # 1️⃣ Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,  # Qwen 官方 tokenizer 主要是正規版
        )

        # 2️⃣ vLLM LLM
        self.llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization=self.quantization,
            gpu_memory_utilization=1.0,
            swap_space=4,
            kv_cache_dtype="fp8",
            max_model_len=self.max_model_len,
        )

        # 3️⃣ 取得 </think> token id（稍後用於切分思考段落）
        self._think_end_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        if isinstance(self._think_end_token_id, list):
            self._think_end_token_id = self._think_end_token_id[0]

    # --------------------------------------------------------------------- #
    # 會話控制
    # --------------------------------------------------------------------- #
    def start_conv(self, system_prompt: Optional[str] = None) -> None:
        """重置歷史，並可選加入 system 提示。"""
        self.conversation = []
        if system_prompt:
            self.conversation.append({"role": "system", "content": system_prompt})

    def end_conv(self) -> None:
        """清空當前對話緩衝。"""
        self.conversation = []

    # --------------------------------------------------------------------- #
    # Sampling 參數管理
    # --------------------------------------------------------------------- #
    def set_default_sampling(self, **kwargs) -> None:
        """更新全局 sampling 預設。"""
        self.default_sampling_kwargs.update(kwargs)

    # --------------------------------------------------------------------- #
    # Prompt 建構
    # --------------------------------------------------------------------- #
    def _build_prompt(self, enable_thinking: bool) -> str:
        """
        生成最終 Prompt，交給 vLLM。
        若 enable_thinking=True，tokenizer 會自動插入 </think> 包裝。
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded – call .load() first.")
        prompt = self.tokenizer.apply_chat_template(
            conversation=self.conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return prompt

    # --------------------------------------------------------------------- #
    # Token‑budget 相關工具
    # --------------------------------------------------------------------- #
    def _available_prompt_tokens(self, gen_cfg: dict) -> int:
        """
        計算本輪可用的 Prompt token 上限。
        max_len：模型最大上下文長度（若未設定則使用 8192）。
        reserve：生成階段保留的 token 數 + 安全邊界。
        """
        max_len = self.max_model_len or 8192
        reserve = gen_cfg.get(
            "max_new_tokens",
            self.default_sampling_kwargs.get("max_new_tokens", 512),
        )
        safety_margin = 50
        return max_len - (reserve + safety_margin)
    '''
    def _message_token_len(self, msg: dict) -> int:
        text = msg["content"]
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    '''

    def _message_token_len(self, msg: dict) -> int:
        text = msg.get("content", "")
        # 1️⃣ 确认类型
        if not isinstance(text, str):
            raise TypeError(f"msg['content'] 需为 str，实际为 {type(text)}")
        # 2️⃣ 打印前 200 个字符的 repr，便于看到不可见字符
        print(">>> content repr:", repr(text[:200]))
        # 3️⃣ 检查是否含有代理字符
        surrogates = [c for c in text if 0xD800 <= ord(c) <= 0xDFFF]
        if surrogates:
            print(f"⚠️ 发现 {len(surrogates)} 个未配对的代理字符，位置示例：{[hex(ord(c)) for c in surrogates[:5]]}")
        # 继续正常编码
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # --------------------------------------------------------------------- #
    # 歷史裁剪與摘要壓縮
    # --------------------------------------------------------------------- #
    def _trim_by_window(self,
                        max_prompt_tokens: int,
                        keep_last_k: Optional[int] = None) -> bool:
        """
        只保留最近 ``keep_last_k`` 轮（user+assistant）以及所有 system 消息。
        若未传入 keep_last_k，则使用实例属性 self.first_trim_k。
        """
        if keep_last_k is None:
            keep_last_k = self.first_trim_k   # ← 这里取默认值

        system_msgs = [m for m in self.conversation if m["role"] == "system"]
        other_msgs   = [m for m in self.conversation if m["role"] != "system"]

        # 把对话切成轮次（每轮两条：user、assistant）
        rounds = [other_msgs[i:i + 2] for i in range(0, len(other_msgs), 2)]
        recent_rounds = rounds[-keep_last_k:]

        trimmed = system_msgs + [msg for r in recent_rounds for msg in r]
        total_len = sum(self._message_token_len(m) for m in trimmed)

        if total_len <= max_prompt_tokens:
            self.conversation = trimmed
            return True
        return False


    def _summarize_prefix(
        self, prefix_msgs: List[dict], max_summary_tokens: int = 200
    ) -> str:
        """
        使用同一本 LLM 為較早的對話生成簡短摘要。
        摘要會以一條 system 訊息的形式回插到歷史中。
        """
        # 1️⃣ 摘要提示
        summary_system_msg = {
            "role": "system",
            "content": f"請對以下對話做簡要摘要，字數不超過 {max_summary_tokens} 個 token。",
        }
        convo_for_summary = [summary_system_msg] + prefix_msgs

        # 2️⃣ 產生 Prompt（關閉思考模式）
        prompt = self.tokenizer.apply_chat_template(
            conversation=convo_for_summary,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 3️⃣ 采樣參數：低溫、限制長度
        summary_params = SamplingParams(
            temperature=0.2,
            max_new_tokens=max_summary_tokens,
            stop=None,
        )

        # 4️⃣ 呼叫 LLM
        outputs = self.llm.generate(
            prompts=[prompt], sampling_params=summary_params
        )
        summary_text = outputs[0].outputs[0].text.strip()
        return summary_text

    def _compress_history(self,
                        max_prompt_tokens: int,
                        keep_last_k: Optional[int] = None) -> None:
        # ① 窗口裁剪
        if self._trim_by_window(max_prompt_tokens, keep_last_k=keep_last_k):
            return

        # ② 準備分割：保留最近 K 輪，其餘做摘要
        system_msgs = [m for m in self.conversation if m["role"] == "system"]
        other_msgs = [m for m in self.conversation if m["role"] != "system"]
        rounds = [other_msgs[i : i + 2] for i in range(0, len(other_msgs), 2)]

        keep_last_k = 6
        suffix_rounds = rounds[-keep_last_k:]   # 最近的 K 輪
        prefix_rounds = rounds[:-keep_last_k]   # 需要被壓縮的部分

        if not prefix_rounds:
            # 沒有可壓縮的前綴，直接硬截斷
            self.conversation = system_msgs + [
                msg for r in suffix_rounds for msg in r
            ]
            return

        # ③ 產生摘要
        prefix_msgs = [msg for r in prefix_rounds for msg in r]
        summary_text = self._summarize_prefix(prefix_msgs, max_summary_tokens=150)

        summary_msg = {
            "role": "system",
            "content": f"以下是之前對話的簡要概述：{summary_text}",
        }

        # ④ 合併摘要與最近輪次
        self.conversation = system_msgs + [summary_msg] + [
            msg for r in suffix_rounds for msg in r
        ]

        # ⑤ 再次檢查，以防仍超限，最後硬截斷最早的訊息（包括摘要）
        total_len = sum(self._message_token_len(m) for m in self.conversation)
        if total_len > max_prompt_tokens:
            # 只保留最近 (keep_last_k*2 + 1) 條訊息（+1 為可能的 system/summary）
            flat = [msg for r in suffix_rounds for msg in r]
            self.conversation = system_msgs + flat[-(keep_last_k * 2 + 1) :]

    # --------------------------------------------------------------------- #
    # 生成 (Chat) 主流程
    # --------------------------------------------------------------------- #
    def chat(
        self,
        user_message: str,
        *,
        enable_thinking: bool = False,
        keep_last_k: Optional[int] = None,   # 新增参数
        **gen_kwargs,
    ) -> Dict[str, Optional[str]]:
        """
        傳入使用者訊息，返回思考（若有）與最終答案。

        Returns
        -------
        dict
            - "thinking": str | None   （若 enable_thinking=True 且模型產生了 </think>）
            - "answer"  : str          （最終答案，不含 </think>）
        """
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Engine not loaded – call .load() first.")

        # 1️⃣ 加入本輪 user 訊息（尚未裁剪）
        self.conversation.append({"role": "user", "content": user_message})

        # 2️⃣ 合併 sampling 參數
        sampling_cfg = self.default_sampling_kwargs.copy()
        sampling_cfg.update(gen_kwargs)

        # 3️⃣ vLLM 只接受 max_tokens，將 max_new_tokens 轉換
        if "max_new_tokens" in sampling_cfg:
            sampling_cfg["max_tokens"] = sampling_cfg.pop("max_new_tokens")

        # 计算 token 预算
        max_prompt_tokens = self._available_prompt_tokens(sampling_cfg)

        # 5️⃣ 壓縮歷史（去除思考、裁剪或摘要），保證 Prompt 在 token 預算內
        self._compress_history(max_prompt_tokens, keep_last_k=keep_last_k)

        # 6️⃣ 構造最終 Prompt（此時已符合 token 限制）
        prompt = self._build_prompt(enable_thinking=enable_thinking)

        # 7️⃣ 生成 SamplingParams 物件
        sampling_params = SamplingParams(**sampling_cfg)

        # 8️⃣ 呼叫 vLLM
        outputs = self.llm.generate(prompts=[prompt], sampling_params=sampling_params)
        request_output = outputs[0]
        generate_output = request_output.outputs[0]

        token_ids: List[int] = generate_output.token_ids
        full_text: str = generate_output.text

        # 9️⃣ 思考段切分（若有開啟）
        thinking_text: Optional[str] = None
        answer_text: str = full_text

        if enable_thinking and self._think_end_token_id is not None:
            try:
                think_end_index = token_ids.index(self._think_end_token_id)
                thinking_ids = token_ids[:think_end_index]
                answer_ids = token_ids[think_end_index:]  # 包含 </think> 本身，後面即答案

                thinking_text = self.tokenizer.decode(
                    thinking_ids, skip_special_tokens=True
                ).strip()
                answer_text = self.tokenizer.decode(
                    answer_ids, skip_special_tokens=True
                ).strip()
            except ValueError:
                # 沒有 </think>，整段直接作為答案
                thinking_text = None
                answer_text = full_text


        # 10️⃣ 把最終答案寫回歷史（思考不寫入）
        self.conversation.append({"role": "assistant", "content": answer_text})

        return {"thinking": thinking_text, "answer": answer_text}

    # --------------------------------------------------------------------- #
    # 其他實用方法
    # --------------------------------------------------------------------- #
    def get_history(self) -> List[Dict[str, str]]:
        """返回當前對話歷史的淺拷貝。"""
        return self.conversation.copy()

    def __del__(self):
        """在物件被回收時嘗試關閉 vLLM 資源。"""
        try:
            if self.llm is not None:
                self.llm.shutdown()
        except Exception:
            pass


# ==============================
# 使用示例（直接在此文件最底部執行）
# ==============================
if __name__ == "__main__":
    # 替換為你本地模型的路徑
    engine = QwenChatEngine("/mnt/e/models/Qwen3-8B_int4")
    engine.load()
    engine.start_conv(system_prompt="You are a helpful assistant.")

    resp = engine.chat(
        "請介紹一下大語言模型的基本概念。",
        enable_thinking=True,        # 開啟 </think>（思考會返回在 resp["thinking"]）
        max_new_tokens=256,
        temperature=0.7,
    )
    print("思考階段:")
    print(resp["thinking"])
    print("\n最終答案:")
    print(resp["answer"])

    engine.end_conv()
