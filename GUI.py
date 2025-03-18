import tkinter as tk
from tkinter import ttk
import subprocess  

class GPTTrainerGUI:
    def __init__(self, master):
        self.master = master
        master.title("GPT_trainer_GUI")
        master.geometry("900x600")
        master.configure(bg="darkgrey")

        self.running_state = False

        # Parameter: max_steps (default to minimum slider value)
        self.max_steps = tk.IntVar(value=300)
        self.k_folds = tk.IntVar(value=5)  

        # fonts
        title_font = ("Helvetica", 16, "bold")
        button_font = ("Helvetica", 12, "bold")
        label_font = ("Helvetica", 12)

        
        self.title_label = tk.Label(master, text="GPT_trainer_GUI", font=title_font, bg="darkgrey", fg="black")
        self.title_label.pack(pady=20)

        # Frame for dropdown boxes and slider in the middle.
        middle_frame = tk.Frame(master, bg="darkgrey")
        middle_frame.pack(pady=40)

        # ----- Model dropdown -----
        model_label = tk.Label(middle_frame, text="Model:", font=label_font, bg="darkgrey", fg="black")
        model_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        
        model_options = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        self.selected_model = tk.StringVar(value=model_options[0])
        model_dropdown = ttk.Combobox(middle_frame, textvariable=self.selected_model, values=model_options, state="readonly")
        model_dropdown.grid(row=0, column=1, padx=10, pady=5)

        # ----- Dataset dropdown -----
        dataset_label = tk.Label(middle_frame, text="Dataset (Taskmaster2 ):", font=label_font, bg="darkgrey", fg="black")
        dataset_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
        
        dataset_options = ["flights", "restaurant-search", "sports", "music", "hotels","food-ordering"]
        self.selected_dataset = tk.StringVar(value=dataset_options[0])
        dataset_dropdown = ttk.Combobox(middle_frame, textvariable=self.selected_dataset, values=dataset_options, state="readonly")
        dataset_dropdown.grid(row=1, column=1, padx=10, pady=5)

        # ----- K-Folds dropdown -----
        k_folds_label = tk.Label(middle_frame, text="K-Folds:", font=label_font, bg="darkgrey", fg="black")
        k_folds_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        

        k_folds_options = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.selected_k_folds = tk.StringVar(value=k_folds_options[0])
        k_folds_dropdown = ttk.Combobox(middle_frame, textvariable=self.selected_k_folds, values=k_folds_options, state="readonly")
        k_folds_dropdown.grid(row=2, column=1, padx=10, pady=5)

        # ----- Max Steps slider -----
        slider_label = tk.Label(middle_frame, text="Max Steps:", font=label_font, bg="darkgrey", fg="black")
        slider_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        
        # Slider: from 10 to 600 
        self.max_steps_slider = tk.Scale(middle_frame, from_=10, to=600, orient=tk.HORIZONTAL,
                                         variable=self.max_steps, bg="darkgrey", fg="black",
                                         font=label_font, length=400,
                                         command=self.update_slider_value)
        self.max_steps_slider.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        # Label to display the current slider value.
        self.max_steps_value_label = tk.Label(middle_frame, text=f"{self.max_steps.get()}", font=label_font, bg="darkgrey", fg="black")
        self.max_steps_value_label.grid(row=3, column=2, padx=10, pady=5)


        # Create a frame for buttons at the bottom
        button_frame = tk.Frame(master, bg="darkgrey")
        button_frame.pack(side=tk.BOTTOM, pady=20)


        # Stop button: red background, and closes the application.
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.on_stop, width=10,
                                     font=button_font, bg="red", fg="white")
        self.stop_button.pack(side=tk.LEFT, padx=10)


        # Run button: green background.
        self.run_button = tk.Button(button_frame, text="Run", command=self.on_run, width=10,
                                    font=button_font, bg="green", fg="white")
        self.run_button.pack(side=tk.LEFT, padx=10)


        # blue generate button
        self.run_best_button = tk.Button(button_frame, text="Generate", command=self.on_run_best, width=10,
                                  font=button_font, bg="blue", fg="white")
        self.run_best_button.pack(side=tk.LEFT, padx=10)



    def on_run_best(self):
        print("Run Best button pressed.")
        
        command = ["python", "run-best.py"]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}")

    def update_slider_value(self, event):
        self.max_steps_value_label.config(text=f"{self.max_steps.get()}")

    def on_stop(self):
        # debug information 
        print("Stop button pressed. Closing the application.")
        print("Selected Model:", self.selected_model.get())
        print("Selected Dataset:", self.selected_dataset.get())
        print("Max Steps:", self.max_steps.get())
        print("K-Folds:", self.selected_k_folds.get())
        self.master.destroy()

    def on_run(self):
        self.running_state = True
        print("Run button pressed. running_state =", self.running_state)
        print("Selected Model:", self.selected_model.get())
        print("Selected Dataset:", self.selected_dataset.get())
        print("Max Steps:", self.max_steps.get())
        print("K-Folds:", self.selected_k_folds.get())


        command = [
            "python", "main.py",
            f"--max_steps={self.max_steps.get()}",
            f"--dataset={self.selected_dataset.get()}",
            f"--model={self.selected_model.get()}",
            f"--k={self.selected_k_folds.get()}"  
        ]

        # Execute
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command: {e}")

def read_and_print_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
            print(content)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print("An error occurred while reading the file:", e)

def main():
    root = tk.Tk()
    app = GPTTrainerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    read_and_print_file("logo.txt")
    main()