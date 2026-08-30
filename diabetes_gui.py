import math
import tkinter as tk
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from tkinter import messagebox

PLACEHOLDER_VALUES = [
    "6",
    "148",
    "72",
    "35",
    "0",
    "33.6",
    "0.627",
    "50",
]

COLORS = {
    "navy": "#12304A",
    "navy_deep": "#0B2235",
    "teal": "#0B8F87",
    "teal_dark": "#08736D",
    "teal_soft": "#DDF3F1",
    "canvas": "#F3F7F9",
    "card": "#FFFFFF",
    "border": "#D9E3E8",
    "text": "#183142",
    "muted": "#627785",
    "subtle": "#8A9BA5",
    "lower": "#178562",
    "lower_soft": "#E0F3EC",
    "elevated": "#C94C5D",
    "elevated_soft": "#FBE8EB",
    "warning": "#B33B4B",
}

FIELD_DETAILS = {
    "Pregnancy": ("Pregnancies", "count"),
    "Glucose": ("Glucose", "mg/dL"),
    "BloodPressure": ("Blood pressure", "mm Hg"),
    "SkinThickness": ("Skin thickness", "mm"),
    "Insulin": ("Insulin", "µU/mL"),
    "BMI": ("Body mass index", "kg/m²"),
    "DiabetesPedigreeFunction": ("Diabetes pedigree", "score"),
    "Age": ("Age", "years"),
}


def parse_inputs(raw_values, feature_labels):
    if len(raw_values) != len(feature_labels):
        raise ValueError("Input and label counts must match.")

    values = []
    for raw_value, label in zip(raw_values, feature_labels):
        raw_value = raw_value.strip()
        if not raw_value:
            return None, f"{label} is required."
        try:
            value = float(raw_value)
        except ValueError:
            return None, f"{label} must be a number."
        if not math.isfinite(value):
            return None, f"{label} must be a finite number."
        values.append(value)
    return values, None


def format_prediction(result):
    if isinstance(result, tuple) and len(result) >= 2:
        label, likelihood = result[:2]
    else:
        label, likelihood = result, None

    if type(label) is not int or label not in (0, 1):
        raise ValueError("Predictor returned an invalid output.")

    if label == 1:
        title = "Elevated indicator"
        tone = "elevated"
    else:
        title = "Lower indicator"
        tone = "lower"

    likelihood_text = "Not available"
    percent = None
    if likelihood is not None:
        percent = float(likelihood)
        if not math.isfinite(percent):
            raise ValueError("Predictor likelihood must be finite.")
        rounded = Decimal(str(likelihood)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        likelihood_text = f"{rounded}%"

    return {
        "title": title,
        "likelihood": likelihood_text,
        "percent": percent,
        "tone": tone,
    }


def window(feature_labels, predict_callback):
    root = tk.Tk()
    root.title("Diabetes Predictor")
    root.geometry("1040x740")
    root.minsize(900, 650)
    root.configure(bg=COLORS["canvas"])

    asset_directory = Path(__file__).resolve().parent / "assets"
    icon_png = asset_directory / "diabetes-predictor-256.png"
    icon_ico = asset_directory / "diabetes-predictor.ico"
    try:
        app_icon = tk.PhotoImage(file=str(icon_png))
        root.iconphoto(True, app_icon)
        root._app_icon = app_icon
    except tk.TclError:
        app_icon = None
    try:
        root.iconbitmap(str(icon_ico))
    except tk.TclError:
        pass

    header = tk.Frame(root, bg=COLORS["navy"], height=138)
    header.pack(fill="x")
    header.pack_propagate(False)

    header_inner = tk.Frame(header, bg=COLORS["navy"])
    header_inner.pack(fill="both", expand=True, padx=34, pady=24)

    if app_icon is not None:
        header_icon = app_icon.subsample(4, 4)
        icon_label = tk.Label(
            header_inner,
            image=header_icon,
            bg=COLORS["navy"],
            borderwidth=0,
        )
        icon_label.image = header_icon
        icon_label.pack(side="left", padx=(0, 18))

    heading_group = tk.Frame(header_inner, bg=COLORS["navy"])
    heading_group.pack(side="left", fill="y")
    tk.Label(
        heading_group,
        text="DIABETES PREDICTOR",
        font=("Segoe UI", 20, "bold"),
        fg="#FFFFFF",
        bg=COLORS["navy"],
    ).pack(anchor="w")
    tk.Label(
        heading_group,
        text="A lightweight blood-glucose risk indicator",
        font=("Segoe UI", 10),
        fg="#C5D8E3",
        bg=COLORS["navy"],
    ).pack(anchor="w", pady=(4, 0))

    body = tk.Frame(root, bg=COLORS["canvas"])
    body.pack(fill="both", expand=True)

    canvas = tk.Canvas(body, bg=COLORS["canvas"], highlightthickness=0)
    scrollbar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    content = tk.Frame(canvas, bg=COLORS["canvas"])
    content_window = canvas.create_window((0, 0), window=content, anchor="nw")

    def sync_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def sync_content_width(event):
        canvas.itemconfigure(content_window, width=event.width)

    content.bind("<Configure>", sync_scroll_region)
    canvas.bind("<Configure>", sync_content_width)

    def scroll_with_wheel(event):
        units = int(-event.delta / 120)
        if units == 0 and event.delta:
            units = -1 if event.delta > 0 else 1
        canvas.yview_scroll(units, "units")

    def scroll_up(_event):
        canvas.yview_scroll(-1, "units")

    def scroll_down(_event):
        canvas.yview_scroll(1, "units")

    dashboard = tk.Frame(content, bg=COLORS["canvas"])
    dashboard.pack(fill="both", expand=True, padx=28, pady=(26, 18))
    dashboard.grid_columnconfigure(0, weight=5, uniform="dashboard")
    dashboard.grid_columnconfigure(1, weight=3, uniform="dashboard")
    dashboard.grid_rowconfigure(0, weight=1)

    form_card = tk.Frame(
        dashboard,
        bg=COLORS["card"],
        padx=26,
        pady=22,
        highlightbackground=COLORS["border"],
        highlightthickness=1,
    )
    form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

    tk.Label(
        form_card,
        text="Patient measurements",
        font=("Segoe UI", 15, "bold"),
        fg=COLORS["text"],
        bg=COLORS["card"],
    ).pack(anchor="w")
    tk.Label(
        form_card,
        text="Enter all eight values. Example values are preloaded for demonstration.",
        font=("Segoe UI", 9),
        fg=COLORS["muted"],
        bg=COLORS["card"],
    ).pack(anchor="w", pady=(4, 18))

    fields = tk.Frame(form_card, bg=COLORS["card"])
    fields.pack(fill="x")
    fields.grid_columnconfigure(0, weight=1, uniform="fields")
    fields.grid_columnconfigure(1, weight=1, uniform="fields")

    entries = []
    entry_borders = []
    for index, raw_label in enumerate(feature_labels):
        display_label, unit = FIELD_DETAILS.get(raw_label, (raw_label, "value"))
        field = tk.Frame(fields, bg=COLORS["card"])
        field.grid(
            row=index // 2,
            column=index % 2,
            sticky="ew",
            padx=(0, 9) if index % 2 == 0 else (9, 0),
            pady=(0, 15),
        )

        label_row = tk.Frame(field, bg=COLORS["card"])
        label_row.pack(fill="x", pady=(0, 5))
        tk.Label(
            label_row,
            text=display_label,
            font=("Segoe UI", 9, "bold"),
            fg=COLORS["text"],
            bg=COLORS["card"],
        ).pack(side="left")
        tk.Label(
            label_row,
            text=unit,
            font=("Segoe UI", 8),
            fg=COLORS["subtle"],
            bg=COLORS["card"],
        ).pack(side="right")

        entry_border = tk.Frame(
            field,
            bg=COLORS["card"],
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["teal"],
            highlightthickness=1,
        )
        entry_border.pack(fill="x")
        entry = tk.Entry(
            entry_border,
            font=("Segoe UI", 11),
            fg=COLORS["text"],
            bg="#FBFDFE",
            insertbackground=COLORS["teal"],
            relief="flat",
            borderwidth=0,
        )
        entry.pack(fill="x", ipady=8, padx=10)
        if index < len(PLACEHOLDER_VALUES):
            entry.insert(0, PLACEHOLDER_VALUES[index])
        entry.bind(
            "<FocusIn>",
            lambda _event, border=entry_border: border.configure(
                highlightbackground=COLORS["teal"], highlightthickness=2
            ),
        )
        entry.bind(
            "<FocusOut>",
            lambda _event, border=entry_border: border.configure(
                highlightbackground=COLORS["border"], highlightthickness=1
            ),
        )
        entries.append(entry)
        entry_borders.append(entry_border)

    form_feedback = tk.StringVar(value="All fields are required before analysis.")
    feedback_label = tk.Label(
        form_card,
        textvariable=form_feedback,
        font=("Segoe UI", 9),
        fg=COLORS["muted"],
        bg=COLORS["card"],
    )
    feedback_label.pack(anchor="w", pady=(0, 14))

    action_row = tk.Frame(form_card, bg=COLORS["card"])
    action_row.pack(fill="x")

    result_card = tk.Frame(
        dashboard,
        bg=COLORS["card"],
        padx=25,
        pady=22,
        highlightbackground=COLORS["border"],
        highlightthickness=1,
    )
    result_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

    tk.Label(
        result_card,
        text="ANALYSIS RESULT",
        font=("Segoe UI", 8, "bold"),
        fg=COLORS["teal_dark"],
        bg=COLORS["card"],
    ).pack(anchor="w")

    status_row = tk.Frame(result_card, bg=COLORS["card"])
    status_row.pack(fill="x", pady=(14, 22))
    status_dot = tk.Label(
        status_row,
        text="●",
        font=("Segoe UI", 13),
        fg=COLORS["subtle"],
        bg=COLORS["card"],
    )
    status_dot.pack(side="left", padx=(0, 8))
    result_title = tk.StringVar(value="Ready for analysis")
    result_title_label = tk.Label(
        status_row,
        textvariable=result_title,
        font=("Segoe UI", 14, "bold"),
        fg=COLORS["text"],
        bg=COLORS["card"],
        wraplength=245,
        justify="left",
    )
    result_title_label.pack(side="left", anchor="w")

    tk.Label(
        result_card,
        text="Indicator likelihood",
        font=("Segoe UI", 9),
        fg=COLORS["muted"],
        bg=COLORS["card"],
    ).pack(anchor="w")
    likelihood_value = tk.StringVar(value="—")
    likelihood_label = tk.Label(
        result_card,
        textvariable=likelihood_value,
        font=("Segoe UI", 28, "bold"),
        fg=COLORS["navy"],
        bg=COLORS["card"],
    )
    likelihood_label.pack(anchor="w", pady=(1, 8))

    gauge = tk.Canvas(
        result_card,
        width=250,
        height=10,
        bg=COLORS["card"],
        highlightthickness=0,
    )
    gauge.pack(fill="x", pady=(0, 19))

    result_detail = tk.StringVar(
        value="Complete the measurements and select Analyze indicator to see a result."
    )
    detail_label = tk.Label(
        result_card,
        textvariable=result_detail,
        font=("Segoe UI", 9),
        fg=COLORS["muted"],
        bg=COLORS["card"],
        wraplength=270,
        justify="left",
    )
    detail_label.pack(anchor="w")

    tk.Label(
        result_card,
        text=(
            "This heuristic is for learning and demonstration only. "
            "It is not a medical diagnosis or a substitute for clinical advice."
        ),
        font=("Segoe UI", 8),
        fg=COLORS["muted"],
        bg=COLORS["card"],
        wraplength=270,
        justify="left",
    ).pack(anchor="w", pady=(18, 0))

    def draw_gauge(percent=None, color=None):
        gauge.delete("all")
        width = max(gauge.winfo_width(), 250)
        gauge.create_rectangle(0, 2, width, 8, fill="#E5ECEF", outline="")
        if percent is not None:
            bounded_percent = max(0.0, min(100.0, percent))
            gauge.create_rectangle(
                0,
                2,
                width * bounded_percent / 100.0,
                8,
                fill=color,
                outline="",
            )

    def reset_result():
        result_title.set("Ready for analysis")
        likelihood_value.set("—")
        result_detail.set(
            "Complete the measurements and select Analyze indicator to see a result."
        )
        status_dot.configure(fg=COLORS["subtle"])
        result_title_label.configure(fg=COLORS["text"])
        likelihood_label.configure(fg=COLORS["navy"])
        result_card.configure(highlightbackground=COLORS["border"])
        draw_gauge()

    def highlight_invalid_entry(raw_values):
        for border in entry_borders:
            border.configure(highlightbackground=COLORS["border"], highlightthickness=1)
        for index, raw_value in enumerate(raw_values):
            invalid = not raw_value.strip()
            if not invalid:
                try:
                    invalid = not math.isfinite(float(raw_value))
                except ValueError:
                    invalid = True
            if invalid:
                entry_borders[index].configure(
                    highlightbackground=COLORS["warning"], highlightthickness=2
                )
                entries[index].focus_set()
                entries[index].selection_range(0, "end")
                break

    def on_predict():
        raw_values = [entry.get() for entry in entries]
        values, error = parse_inputs(raw_values, feature_labels)
        if error is not None:
            form_feedback.set(error)
            feedback_label.configure(fg=COLORS["warning"])
            highlight_invalid_entry(raw_values)
            return
        for border in entry_borders:
            border.configure(highlightbackground=COLORS["border"], highlightthickness=1)

        try:
            presentation = format_prediction(predict_callback(values))
        except Exception as exc:
            form_feedback.set("Unable to calculate a result from these measurements.")
            feedback_label.configure(fg=COLORS["warning"])
            messagebox.showerror(
                "Prediction error",
                f"The indicator could not be calculated.\n\n{exc}",
                parent=root,
            )
            return

        form_feedback.set("Measurements accepted. Result updated.")
        feedback_label.configure(fg=COLORS["lower"])
        is_elevated = presentation["tone"] == "elevated"
        color = COLORS["elevated"] if is_elevated else COLORS["lower"]
        result_title.set(presentation["title"])
        likelihood_value.set(presentation["likelihood"])
        result_detail.set(
            "The score is above this dataset-derived decision threshold. "
            "Discuss real health concerns with a qualified clinician."
            if is_elevated
            else "The score is below this dataset-derived decision threshold. "
            "This does not rule out diabetes or replace clinical testing."
        )
        status_dot.configure(fg=color)
        result_title_label.configure(fg=color)
        likelihood_label.configure(fg=color)
        result_card.configure(highlightbackground=color)
        draw_gauge(presentation["percent"], color)

    def clear_form():
        for entry in entries:
            entry.delete(0, "end")
        for border in entry_borders:
            border.configure(highlightbackground=COLORS["border"], highlightthickness=1)
        form_feedback.set("Form cleared. All fields are required before analysis.")
        feedback_label.configure(fg=COLORS["muted"])
        reset_result()
        entries[0].focus_set()

    analyze_button = tk.Button(
        action_row,
        text="Analyze indicator",
        command=on_predict,
        font=("Segoe UI", 10, "bold"),
        fg="#FFFFFF",
        bg=COLORS["teal"],
        activeforeground="#FFFFFF",
        activebackground=COLORS["teal_dark"],
        relief="flat",
        borderwidth=0,
        cursor="hand2",
        padx=20,
        pady=10,
    )
    analyze_button.pack(side="left")

    clear_button = tk.Button(
        action_row,
        text="Clear form",
        command=clear_form,
        font=("Segoe UI", 10),
        fg=COLORS["navy"],
        bg="#EDF3F6",
        activeforeground=COLORS["navy"],
        activebackground="#E1EAEF",
        relief="flat",
        borderwidth=0,
        cursor="hand2",
        padx=17,
        pady=10,
    )
    clear_button.pack(side="left", padx=(10, 0))

    tk.Label(
        content,
        text="Enter  Analyze indicator    •    Esc  Clear form",
        font=("Segoe UI", 8),
        fg=COLORS["subtle"],
        bg=COLORS["canvas"],
    ).pack(anchor="center", pady=(0, 18))

    def descendants_of(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants_of(child)

    root.bind("<Return>", lambda _event: on_predict())
    root.bind("<Escape>", lambda _event: clear_form())
    for widget in (canvas, content, *tuple(descendants_of(content))):
        widget.bind("<MouseWheel>", scroll_with_wheel, add="+")
        widget.bind("<Button-4>", scroll_up, add="+")
        widget.bind("<Button-5>", scroll_down, add="+")
    root.after_idle(draw_gauge)

    root.mainloop()
