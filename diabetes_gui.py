import tkinter as tk
from tkinter import messagebox


def window(feature_labels, predict_callback):
    root = tk.Tk()
    root.title("Diabetes Predictor")
    root.geometry("520x460")
    root.resizable(False, False)

    container = tk.Frame(root, padx=16, pady=16)
    container.pack(fill="both", expand=True)

    title = tk.Label(container, text="Enter Patient Features", font=("Segoe UI", 14, "bold"))
    title.pack(anchor="w", pady=(0, 10))

    form = tk.Frame(container)
    form.pack(fill="x", pady=(0, 12))

    entries = []
    row_index = 0
    while row_index < len(feature_labels):
        label = tk.Label(form, text=feature_labels[row_index] + ":", width=24, anchor="w")
        label.grid(row=row_index, column=0, sticky="w", pady=4)

        entry = tk.Entry(form, width=24)
        entry.grid(row=row_index, column=1, sticky="w", pady=4)
        entries.append(entry)
        row_index += 1

    output_value = tk.StringVar(value="-")
    likelihood_value = tk.StringVar(value="-")

    def on_predict():
        values = []
        i = 0
        while i < len(entries):
            raw = entries[i].get().strip()
            if raw == "":
                messagebox.showwarning("Invalid Input", feature_labels[i] + " is required.")
                return
            try:
                values.append(float(raw))
            except Exception:
                messagebox.showwarning("Invalid Input", feature_labels[i] + " must be numeric.")
                return
            i += 1

        result = predict_callback(values)

        y = None
        likelihood = None

        if isinstance(result, tuple) and len(result) >= 2:
            y = result[0]
            likelihood = result[1]
        else:
            y = result

        if y != 0 and y != 1:
            messagebox.showerror("Prediction Error", "Predictor returned an invalid output.")
            return

        output_value.set(str(y))

        if likelihood is None:
            likelihood_value.set("-")
        else:
            likelihood_value.set("{0:.2f}%".format(float(likelihood)))

    button = tk.Button(container, text="Predict", width=14, command=on_predict)
    button.pack(anchor="w", pady=(0, 12))

    result_row = tk.Frame(container)
    result_row.pack(fill="x")
    tk.Label(result_row, text="Prediction (1 or 0):", font=("Segoe UI", 11, "bold")).pack(side="left")
    tk.Label(result_row, textvariable=output_value, font=("Segoe UI", 11)).pack(side="left", padx=(8, 0))

    context_row = tk.Frame(container)
    context_row.pack(fill="x", pady=(8, 0))
    tk.Label(context_row, text="Likelihood of diabetes:", font=("Segoe UI", 11, "bold")).pack(side="left")
    tk.Label(context_row, textvariable=likelihood_value, font=("Segoe UI", 11)).pack(side="left", padx=(8, 0))

    hint_row = tk.Frame(container)
    hint_row.pack(fill="x", pady=(6, 0))
    tk.Label(
        hint_row,
        text="Context: 1 = above threshold risk, 0 = below threshold risk",
        font=("Segoe UI", 9),
    ).pack(anchor="w")

    root.mainloop()
