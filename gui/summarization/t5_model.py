import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5Model:
    def __init__(self):
        """
        初始化 T5 模型和设备配置
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "../model/t5_finetuned")  # 替换为你的模型路径或模型 ID

        try:
            print(f"Loading tokenizer from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("Tokenizer loaded successfully.")

            print(f"Loading model from: {self.model_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            print("Model loaded successfully.")

            # 检查是否支持 MPS
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS backend for model inference.")
            else:
                self.device = torch.device("cpu")
                print("MPS not available. Using CPU.")

            # 将模型移动到设备
            self.model.to(self.device)
            print(f"Model moved to device: {self.device}")

        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise

    def summarize(self, text, length):
        """
        使用 T5 模型生成摘要
        """
        try:
            print("Starting summarization process...")
            print(f"Input text: {text[:100]}")  # 打印输入文本的前 100 个字符

            # 编码输入文本
            input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
            print(f"Encoded input IDs: {input_ids}")
            print(f"Input IDs size: {input_ids.size()}")  # 调试打印

            # 检查输入是否为空或过短
            if input_ids.size(1) == 0:
                print("Error: Input text is empty after tokenization.")
                return "Error: Input text is empty after tokenization."
            if input_ids.size(1) < 5:
                print("Error: Input text is too short for summarization.")
                return "Error: Input text is too short for summarization."

            # 将输入张量移动到设备
            input_ids = input_ids.to("cpu")  # 强制使用 CPU
            self.model.to("cpu")

            # 生成摘要
            print("Generating summary...")
            summary_ids = self.model.generate(
                input_ids=input_ids,
                max_length=min(input_ids.size(1) + 50, 512),  # 限制生成摘要的最大长度
                min_length=max(10, input_ids.size(1) // 4),   # 限制生成摘要的最小长度
                length_penalty=1.0,
                num_beams=4,
                num_return_sequences=1,
                early_stopping=True
            )
            print(f"Generated summary IDs: {summary_ids}")

            # 解码生成的摘要
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            print(f"Decoded summary: {summary}")

            # 恢复模型到原设备
            self.model.to(self.device)

            return summary

        except Exception as e:
            print(f"Error during summarization: {e}")
            return f"Error during summarization: {e}"
