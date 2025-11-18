import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import json
import os

class CIPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI-Powered CI/CD Optimizer")
        self.root.geometry("600x650")
        self.root.configure(bg="#f0f0f0")
        
        # --- State Variables ---
        self.model = None
        self.features_list = []
        
        # --- Load Model Artifacts ---
        self.load_assets()
        
        # --- UI Setup ---
        self.create_styles()
        self.create_header()
        self.create_input_form()
        self.create_result_area()
        
        # --- Footer ---
        ttk.Label(root, text="Built with Python, Scikit-Learn & Tkinter", 
                  foreground="#888").pack(side="bottom", pady=10)

    def load_assets(self):
        """Loads the trained model and features list."""
        try:
            if not os.path.exists('ci_model.pkl') or not os.path.exists('model_features.json'):
                messagebox.showerror("Error", "Model files not found!\nPlease run 'train_model.py' first.")
                self.root.destroy()
                return
                
            self.model = joblib.load('ci_model.pkl')
            with open('model_features.json', 'r') as f:
                self.features_list = json.load(f)
            print("✅ Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def create_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Styles for Frame, Labels, and Buttons
        style.configure("Card.TFrame", background="white", relief="raised")
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), background="#f0f0f0", foreground="#333")
        style.configure("TLabel", font=("Helvetica", 11), background="white", foreground="#555")
        style.configure("Result.TLabel", font=("Helvetica", 14, "bold"), background="white")
        
        style.configure("Action.TButton", font=("Helvetica", 12, "bold"), padding=10)
        style.map("Action.TButton", background=[('active', '#0056b3')], foreground=[('active', 'white')])

    def create_header(self):
        header_frame = ttk.Frame(self.root, padding="20 20 20 10")
        header_frame.pack(fill="x")
        
        title = ttk.Label(header_frame, text="Predictive Pipeline Dashboard", style="Header.TLabel")
        title.pack()
        
        subtitle = ttk.Label(header_frame, text="Simulate commits to test AIOps failure prediction", 
                             background="#f0f0f0", foreground="#666")
        subtitle.pack()

    def create_input_form(self):
        # Main container resembling a card
        form_frame = ttk.Frame(self.root, style="Card.TFrame", padding=20)
        form_frame.pack(fill="x", padx=20, pady=10)
        
        # --- Input Fields Grid ---
        
        # 1. Author
        ttk.Label(form_frame, text="Commit Author:").grid(row=0, column=0, sticky="w", pady=5)
        self.author_var = tk.StringVar(value="main_dev")
        author_cb = ttk.Combobox(form_frame, textvariable=self.author_var, state="readonly")
        author_cb['values'] = ('main_dev', 'new_dev', 'contractor')
        author_cb.grid(row=0, column=1, sticky="ew", pady=5, padx=10)
        
        # 2. Dominant File Type
        ttk.Label(form_frame, text="Primary File Type:").grid(row=1, column=0, sticky="w", pady=5)
        self.file_type_var = tk.StringVar(value=".py")
        file_type_cb = ttk.Combobox(form_frame, textvariable=self.file_type_var, state="readonly")
        file_type_cb['values'] = ('.py', '.js', '.md', '.yml', '.css')
        file_type_cb.grid(row=1, column=1, sticky="ew", pady=5, padx=10)
        
        # 3. Files Changed
        ttk.Label(form_frame, text="Files Changed:").grid(row=2, column=0, sticky="w", pady=5)
        self.files_changed_var = tk.IntVar(value=3)
        ttk.Spinbox(form_frame, from_=1, to=100, textvariable=self.files_changed_var).grid(row=2, column=1, sticky="ew", pady=5, padx=10)
        
        # 4. Lines Added
        ttk.Label(form_frame, text="Lines Added:").grid(row=3, column=0, sticky="w", pady=5)
        self.lines_added_var = tk.IntVar(value=20)
        ttk.Entry(form_frame, textvariable=self.lines_added_var).grid(row=3, column=1, sticky="ew", pady=5, padx=10)

        # 5. Lines Deleted
        ttk.Label(form_frame, text="Lines Deleted:").grid(row=4, column=0, sticky="w", pady=5)
        self.lines_deleted_var = tk.IntVar(value=5)
        ttk.Entry(form_frame, textvariable=self.lines_deleted_var).grid(row=4, column=1, sticky="ew", pady=5, padx=10)

        # --- Predict Button ---
        predict_btn = ttk.Button(form_frame, text="🔍 Analyze Commit Risk", style="Action.TButton", command=self.run_prediction)
        predict_btn.grid(row=5, column=0, columnspan=2, pady=20, sticky="ew")
        
        # Configure grid weights
        form_frame.columnconfigure(1, weight=1)

    def create_result_area(self):
        self.result_frame = ttk.Frame(self.root, style="Card.TFrame", padding=20)
        self.result_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.status_label = tk.Label(self.result_frame, text="Ready to Predict", 
                                     font=("Helvetica", 18, "bold"), bg="white", fg="#888")
        self.status_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(self.result_frame, text="", style="Result.TLabel")
        self.confidence_label.pack()
        
        self.action_label = ttk.Label(self.result_frame, text="", font=("Courier", 12), background="white")
        self.action_label.pack(pady=15)

    def prepare_input_data(self, commit_data):
        """Exact logic from run_ci_pipeline.py to ensure consistency"""
        df = pd.DataFrame(commit_data, index=[0])
        df_processed = pd.get_dummies(df)
        df_final = df_processed.reindex(columns=self.features_list, fill_value=0)
        return df_final

    def run_prediction(self):
        # 1. Gather Input
        input_data = {
            'author': self.author_var.get(),
            'files_changed': self.files_changed_var.get(),
            'lines_added': self.lines_added_var.get(),
            'lines_deleted': self.lines_deleted_var.get(),
            'dominant_file_type': self.file_type_var.get()
        }
        
        # 2. Process Data
        processed_data = self.prepare_input_data(input_data)
        
        # 3. Predict
        prediction = self.model.predict(processed_data)[0]
        prediction_proba = self.model.predict_proba(processed_data)[0]
        
        # 4. Update UI
        self.update_result_ui(prediction, prediction_proba)

    def update_result_ui(self, prediction, proba):
        if prediction == 1: # Failure
            self.status_label.config(text="PREDICTION: FAILURE", fg="#d9534f") # Red
            confidence = proba[1]
            action_text = "Pipeline Logic: STOP (Fail Fast)\nSkipping tests to save resources."
        else: # Success
            self.status_label.config(text="PREDICTION: SUCCESS", fg="#28a745") # Green
            confidence = proba[0]
            action_text = "Pipeline Logic: CONTINUE\nRunning full test suite..."
            
        self.confidence_label.config(text=f"AI Confidence: {confidence:.2%}")
        self.action_label.config(text=action_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = CIPredictorApp(root)
    root.mainloop()