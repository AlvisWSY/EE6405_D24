import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from summarization.textrank import textrank_summary
from summarization.lsa import lsa_summary
from summarization.hits import hits_summary
from summarization.t5_model import T5Model
from summarization.preprocess import preprocess_text


class SummarizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLP Summarization Tool")
        self.history = []
        self.t5_model = T5Model()

        # GUI 布局
        self.setup_gui()

    def setup_gui(self):
        """
        初始化 GUI 界面。
        """
        # 设置窗口大小和背景色
        self.root.geometry("900x600")
        self.root.configure(bg="#ffffff")  # 背景色

        # 配置样式
        style = ttk.Style()
        style.theme_use('clam')  # 使用现代样式
        style.configure('TFrame', background='#ffffff')
        style.configure('TLabel', background='#ffffff', font=('Helvetica', 14))
        style.configure('TButton', font=('Helvetica', 12), foreground='#ffffff', background='#007ACC')
        style.map('TButton', background=[('active', '#005F9E')])
        style.configure('TEntry', font=('Helvetica', 12))
        style.configure('TCombobox', font=('Helvetica', 12))
        style.map('TCombobox', fieldbackground=[('readonly', '#ffffff')], background=[('readonly', '#ffffff')])
        style.configure('TLabelframe', background='#ffffff', font=('Helvetica', 12, 'bold'))
        style.configure('TLabelframe.Label', background='#ffffff', font=('Helvetica', 12, 'bold'), foreground='#007ACC')

        # 左侧面板
        left_frame = ttk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 输入区域
        instruction_label = ttk.Label(left_frame, text="输入文本或上传文件以生成摘要（任选其一）：", anchor="w", justify="left")
        instruction_label.pack(fill=tk.X, padx=5, pady=5)

        input_frame = ttk.LabelFrame(left_frame, text="输入区域", padding=(10, 10))
        input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.input_text = tk.Text(input_frame, height=10, font=("Courier", 12))
        self.input_text.pack(fill=tk.BOTH, expand=True)

        upload_frame = ttk.Frame(input_frame)
        upload_frame.pack(fill=tk.X, pady=5)
        upload_label = ttk.Label(upload_frame, text="或上传文件：")
        upload_label.pack(side=tk.LEFT, padx=5)
        upload_button = ttk.Button(upload_frame, text="上传文件", command=self.upload_file)
        upload_button.pack(side=tk.LEFT, padx=5)

        # 设置区域
        interaction_frame = ttk.LabelFrame(left_frame, text="设置", padding=(10, 10))
        interaction_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        method_label = ttk.Label(interaction_frame, text="选择方法：")
        method_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.method_var = tk.StringVar(value="TextRank")
        method_menu = ttk.Combobox(interaction_frame, textvariable=self.method_var, state="readonly")
        method_menu['values'] = ("TextRank", "TF-IDF+LSA", "HITS", "T5")
        method_menu.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        length_label = ttk.Label(interaction_frame, text="摘要长度（句子数）：")
        length_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.length_entry = ttk.Entry(interaction_frame, width=10)
        self.length_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        self.length_entry.insert(0, "3")

        generate_button = ttk.Button(interaction_frame, text="生成摘要", command=self.generate_summary)
        generate_button.grid(row=0, column=4, padx=5, pady=5, sticky='w')

        # 输出区域
        output_frame = ttk.LabelFrame(left_frame, text="输出区域", padding=(10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.output_text = tk.Text(output_frame, height=10, state=tk.DISABLED, font=("Courier", 12))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # 右侧面板
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        history_label = ttk.Label(right_frame, text="摘要历史", font=("Helvetica", 14, "bold"))
        history_label.pack(anchor="n", pady=5)
        self.history_listbox = tk.Listbox(right_frame, font=("Courier", 12))
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        self.history_listbox.bind("<<ListboxSelect>>", self.display_history)

    def upload_file(self):
        """
        上传文件并显示到输入区域。
        """
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("Doc files", "*.doc")])
        if file_path:
            try:
                with open(file_path, 'r', encoding="utf-8") as file:
                    content = file.read()
                    self.input_text.delete(1.0, tk.END)
                    self.input_text.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("错误", f"无法读取文件：{e}")

    def generate_summary(self):
        """
        调用摘要方法并显示摘要结果。
        """
        input_text = self.input_text.get(1.0, tk.END).strip()
        method = self.method_var.get()
        length = self.length_entry.get()

        if not input_text:
            messagebox.showwarning("警告", "请输入文本或上传文件！")
            return
        if not length.isdigit() or int(length) <= 0:
            messagebox.showwarning("警告", "摘要长度必须为正整数！")
            return

        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)

        try:
            summary = self.perform_summarization(input_text, method, int(length))
            self.output_text.insert(tk.END, summary)
            self.output_text.config(state=tk.DISABLED)
            self.history.append((method, input_text[:100], summary))
            self.history_listbox.insert(tk.END, f"{method}: {input_text[:50]}...")
        except Exception as e:
            messagebox.showerror("错误", f"摘要生成失败：{e}")

    def perform_summarization(self, text, method, length):
        """
        根据选择的方法生成摘要。
        """
        if method == "TextRank":
            return textrank_summary(text, length)
        elif method == "TF-IDF+LSA":
            return lsa_summary(text, length)
        elif method == "HITS":
            return hits_summary(text, length)
        elif method == "T5":
            return self.t5_model.summarize(text, length)
        else:
            return "未实现的摘要方法"

    def display_history(self, event):
        """
        显示历史摘要结果。
        """
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            _, _, summary = self.history[index]
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, summary)
            self.output_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = SummarizationApp(root)
    root.mainloop()
