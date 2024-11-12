import tkinter as tk
from tkinter import ttk
from gui import SummarizationApp

if __name__ == "__main__":
    root = tk.Tk()
    app = SummarizationApp(root)

    # 定义 Hover 和 Selected 的样式
    style = ttk.Style()
    style.configure('Hover.TFrame', background='#e6e6e6')
    style.configure('Selected.TFrame', background='#d9d9d9')
    style.configure('TFrame', background='#ffffff')

    root.mainloop()
