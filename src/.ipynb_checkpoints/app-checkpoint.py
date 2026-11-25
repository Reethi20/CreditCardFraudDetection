import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import joblib
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FraudDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Credit Card Fraud Detection")
        self.geometry("900x700")

        # Load model and scaler
        try:
            self.model = joblib.load('models/fraud_detection_model.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
        except FileNotFoundError:
            messagebox.showerror("Error", "Model or scaler not found. Please run the training script first.")
            self.destroy()
            return

        # --- GUI Layout ---
        # Top Frame for button
        top_frame = tk.Frame(self, pady=10)
        top_frame.pack(fill=tk.X)

        self.load_button = tk.Button(top_frame, text="Load Transaction Data (CSV)", command=self.load_and_predict)
        self.load_button.pack()

        # Summary Frame
        summary_frame = tk.Frame(self, padx=10)
        summary_frame.pack(fill=tk.X, side=tk.TOP, pady=(5,0))
        self.summary_label = tk.Label(summary_frame, text="Summary: Load a file to see details.", font=("Arial", 12, "bold"), justify=tk.LEFT)
        self.summary_label.pack(anchor='w')

        # Main Frame for graph and results table
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Graph Frame
        graph_frame = tk.Frame(main_frame)
        graph_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.fig = Figure(figsize=(5, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack()

        # Results Table
        table_frame = tk.Frame(main_frame)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(table_frame, show='headings')
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(self, text="Please load a file to begin.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            self.status_label.config(text=f"Loaded {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return

        # Preprocess data
        df_processed = self.preprocess(df.copy())

        # Predict
        predictions = self.model.predict(df_processed)
        df['Prediction'] = ['Fraud' if p == 1 else 'Not Fraud' for p in predictions]

        self.display_results(df)
        self.update_graph(df['Prediction'])

    def preprocess(self, df):
        # Ensure required columns are present
        if 'Time' not in df.columns or 'Amount' not in df.columns:
            messagebox.showerror("Error", "CSV must contain 'Time' and 'Amount' columns.")
            return None

        df['scaled_amount'] = self.scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = self.scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        df_processed = df.drop(['Time', 'Amount'], axis=1)
        # Ensure columns match model's expected input
        df_processed = df_processed.reindex(columns=self.model.feature_names_in_, fill_value=0)
        return df_processed

    def display_results(self, df):
        # Clear previous results
        for i in self.tree.get_children():
            self.tree.delete(i)

        # Setup columns
        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        # Insert data
        self.tree.tag_configure('fraud', background='red', foreground='white')
        for index, row in df.iterrows():
            tag = 'fraud' if row['Prediction'] == 'Fraud' else ''
            self.tree.insert("", "end", values=list(row), tags=(tag,))

    def update_graph(self, predictions):
        self.ax.clear()
        counts = predictions.value_counts()
        labels = counts.index
        sizes = counts.values
        colors = ['red' if label == 'Fraud' else 'green' for label in labels]

        self.ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        self.fig.tight_layout()
        self.canvas.draw()

        fraud_count = counts.get('Fraud', 0)
        total_count = len(predictions)
        self.summary_label.config(text=f"Summary: Total Transactions: {total_count} | Fraud Detected: {fraud_count}")

if __name__ == "__main__":
    app = FraudDetectionApp()
    app.mainloop()
