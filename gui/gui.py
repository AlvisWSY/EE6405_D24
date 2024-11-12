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

        # GUI Layout
        self.setup_gui()

    def setup_gui(self):
        """
        Initialize the GUI interface.
        """
        # Set window size and background color
        self.root.geometry("1100x600")
        self.root.configure(bg="#ffffff")  # Background color

        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')  # Use modern theme
        style.configure('TFrame', background='#ffffff')
        style.configure('TLabel', background='#ffffff', font=('Helvetica', 14))
        style.configure('TButton', font=('Helvetica', 12), foreground='#ffffff', background='#007ACC')
        style.map('TButton', background=[('active', '#005F9E')])
        style.configure('TEntry', font=('Helvetica', 12))
        style.configure('TCombobox', font=('Helvetica', 12))
        style.map('TCombobox', fieldbackground=[('readonly', '#ffffff')], background=[('readonly', '#ffffff')])
        style.configure('TLabelframe', background='#ffffff', font=('Helvetica', 12, 'bold'))
        style.configure('TLabelframe.Label', background='#ffffff', font=('Helvetica', 12, 'bold'), foreground='#007ACC')

        # Left panel
        left_frame = ttk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Input area
        instruction_label = ttk.Label(left_frame, text="Enter text or upload a file to generate a summary (choose either):", anchor="w", justify="left")
        instruction_label.pack(fill=tk.X, padx=5, pady=5)

        input_frame = ttk.LabelFrame(left_frame, text="Input Area", padding=(10, 10))
        input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.input_text = tk.Text(input_frame, height=10, font=("Courier", 12))
        self.input_text.pack(fill=tk.BOTH, expand=True)

        upload_frame = ttk.Frame(input_frame)
        upload_frame.pack(fill=tk.X, pady=5)
        upload_label = ttk.Label(upload_frame, text="Or upload a file:")
        upload_label.pack(side=tk.LEFT, padx=5)
        upload_button = ttk.Button(upload_frame, text="Upload File", command=self.upload_file)
        upload_button.pack(side=tk.LEFT, padx=5)

        # Settings area
        interaction_frame = ttk.LabelFrame(left_frame, text="Settings", padding=(10, 10))
        interaction_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        method_label = ttk.Label(interaction_frame, text="Select Method:")
        method_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.method_var = tk.StringVar(value="TextRank")
        method_menu = ttk.Combobox(interaction_frame, textvariable=self.method_var, state="readonly")
        method_menu['values'] = ("TextRank", "TF-IDF+LSA", "HITS", "T5")
        method_menu.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        length_label = ttk.Label(interaction_frame, text="Summary Length (Number of Sentences):")
        length_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.length_entry = ttk.Entry(interaction_frame, width=10)
        self.length_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        self.length_entry.insert(0, "3")

        generate_button = ttk.Button(interaction_frame, text="Generate Summary", command=self.generate_summary)
        generate_button.grid(row=0, column=4, padx=5, pady=5, sticky='w')

        # Output area
        output_frame = ttk.LabelFrame(left_frame, text="Output Area", padding=(10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.output_text = tk.Text(output_frame, height=10, state=tk.DISABLED, font=("Courier", 12))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Right panel
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        history_label = ttk.Label(right_frame, text="Summary History", font=("Helvetica", 14, "bold"))
        history_label.pack(anchor="n", pady=5)
        self.history_listbox = tk.Listbox(right_frame, font=("Courier", 12))
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        self.history_listbox.bind("<<ListboxSelect>>", self.display_history)

    def upload_file(self):
        """
        Upload a file and display its content in the input area.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("Doc files", "*.doc")])
        if file_path:
            try:
                with open(file_path, 'r', encoding="utf-8") as file:
                    content = file.read()
                    self.input_text.delete(1.0, tk.END)
                    self.input_text.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("Error", f"Unable to read file: {e}")

    def generate_summary(self):
        """
        Generate a summary using the selected method and display the result.
        """
        input_text = self.input_text.get(1.0, tk.END).strip()
        method = self.method_var.get()
        length = self.length_entry.get()

        if not input_text:
            messagebox.showwarning("Warning", "Please enter text or upload a file!")
            return
        if not length.isdigit() or int(length) <= 0:
            messagebox.showwarning("Warning", "Summary length must be a positive integer!")
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
            messagebox.showerror("Error", f"Failed to generate summary: {e}")

    def perform_summarization(self, text, method, length):
        """
        Generate a summary based on the selected method.
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
            return "Method not implemented"

    def display_history(self, event):
        """
        Display a summary from the history.
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