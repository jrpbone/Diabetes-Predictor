import ctypes
import math
import sys
import tkinter as tk
import unittest

import diabetes_gui as gui
from diabetes_gui import format_prediction, parse_inputs


FEATURE_LABELS = [
    "Pregnancy",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def build_test_window(callback):
    original_tk = gui.tk.Tk
    holder = {}

    class TestTk(original_tk):
        def __init__(self):
            super().__init__()
            self.withdraw()
            holder["root"] = self

        def mainloop(self, *_args, **_kwargs):
            self.update_idletasks()

    gui.tk.Tk = TestTk
    try:
        gui.window(FEATURE_LABELS, callback)
    finally:
        gui.tk.Tk = original_tk
    return holder["root"]


class ParseInputsTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["Glucose", "BMI"]

    def test_converts_numeric_input_to_floats(self):
        values, error = parse_inputs([" 148 ", "33.6"], self.labels)

        self.assertEqual([148.0, 33.6], values)
        self.assertIsNone(error)

    def test_reports_the_first_empty_field(self):
        values, error = parse_inputs(["", "33.6"], self.labels)

        self.assertIsNone(values)
        self.assertEqual("Glucose is required.", error)

    def test_reports_the_first_non_numeric_field(self):
        values, error = parse_inputs(["high", "33.6"], self.labels)

        self.assertIsNone(values)
        self.assertEqual("Glucose must be a number.", error)

    def test_rejects_non_finite_values(self):
        for raw_value in ("nan", "inf", "-inf"):
            with self.subTest(raw_value=raw_value):
                values, error = parse_inputs([raw_value, "33.6"], self.labels)

                self.assertIsNone(values)
                self.assertEqual("Glucose must be a finite number.", error)

    def test_rejects_mismatched_input_and_label_counts(self):
        with self.assertRaisesRegex(ValueError, "counts must match"):
            parse_inputs(["148"], self.labels)


class FormatPredictionTests(unittest.TestCase):
    def test_formats_an_elevated_indicator(self):
        result = format_prediction((1, 72.345))

        self.assertEqual("Elevated indicator", result["title"])
        self.assertEqual("72.35%", result["likelihood"])
        self.assertEqual(72.345, result["percent"])
        self.assertEqual("elevated", result["tone"])

    def test_formats_a_lower_indicator_without_likelihood(self):
        result = format_prediction(0)

        self.assertEqual("Lower indicator", result["title"])
        self.assertEqual("Not available", result["likelihood"])
        self.assertIsNone(result["percent"])
        self.assertEqual("lower", result["tone"])

    def test_rejects_an_invalid_prediction_label(self):
        with self.assertRaisesRegex(ValueError, "invalid output"):
            format_prediction((2, 50))

    def test_rejects_boolean_prediction_labels(self):
        with self.assertRaisesRegex(ValueError, "invalid output"):
            format_prediction((True, 50))

    def test_rejects_non_finite_likelihoods(self):
        for likelihood in (math.nan, math.inf, -math.inf):
            with self.subTest(likelihood=likelihood):
                with self.assertRaisesRegex(ValueError, "likelihood must be finite"):
                    format_prediction((1, likelihood))


class WindowInteractionTests(unittest.TestCase):
    def setUp(self):
        self.received_values = []
        try:
            self.root = build_test_window(self.predict)
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")
        self.widgets = list(descendants(self.root))

    def tearDown(self):
        if hasattr(self, "root") and self.root.winfo_exists():
            self.root.destroy()

    def predict(self, values):
        self.received_values.append(values)
        return 1, 72.345

    def find_button(self, text):
        return next(
            widget
            for widget in self.widgets
            if widget.winfo_class() == "Button" and widget.cget("text") == text
        )

    def test_analyze_button_invokes_the_callback_with_numeric_values(self):
        self.find_button("Analyze indicator").invoke()

        self.assertEqual(
            [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0],
            self.received_values[-1],
        )

    def test_clear_button_empties_all_measurement_entries(self):
        self.find_button("Clear form").invoke()
        entries = [widget for widget in self.widgets if widget.winfo_class() == "Entry"]

        self.assertEqual([""] * 8, [entry.get() for entry in entries])

    def test_invalid_measurement_is_highlighted_without_calling_predictor(self):
        entries = [widget for widget in self.widgets if widget.winfo_class() == "Entry"]
        entries[0].delete(0, "end")

        self.find_button("Analyze indicator").invoke()
        self.root.update_idletasks()

        self.assertEqual([], self.received_values)
        self.assertEqual(
            gui.COLORS["warning"], entries[0].master.cget("highlightbackground")
        )

    def test_keyboard_and_cross_platform_scroll_bindings_are_configured(self):
        self.assertTrue(self.root.bind("<Return>"))
        self.assertTrue(self.root.bind("<Escape>"))
        canvases = [widget for widget in self.widgets if widget.winfo_class() == "Canvas"]
        scroll_canvas = next(canvas for canvas in canvases if canvas.cget("yscrollcommand"))

        self.assertTrue(scroll_canvas.bind("<MouseWheel>"))
        self.assertTrue(scroll_canvas.bind("<Button-4>"))
        self.assertTrue(scroll_canvas.bind("<Button-5>"))

    def test_header_and_disclaimer_omit_clutter_labels(self):
        label_texts = [
            widget.cget("text")
            for widget in self.widgets
            if widget.winfo_class() == "Label"
        ]

        self.assertNotIn("  EDUCATIONAL TOOL  ", label_texts)
        self.assertNotIn("Important", label_texts)
        self.assertTrue(
            any(text.startswith("This heuristic is for learning") for text in label_texts)
        )

    @unittest.skipUnless(sys.platform == "win32", "Windows title-bar icon check")
    def test_windows_title_bar_uses_the_custom_app_icon(self):
        self.root.deiconify()
        self.root.update()
        client_handle = self.root.winfo_id()
        window_handle = ctypes.windll.user32.GetParent(client_handle)
        wm_geticon = 0x007F
        icon_small = 0
        configured_icon = ctypes.windll.user32.SendMessageW(
            window_handle, wm_geticon, icon_small, 0
        )

        self.assertTrue(configured_icon)

    def test_keyboard_shortcuts_dispatch_analyze_and_clear_actions(self):
        self.root.deiconify()
        self.root.update()
        self.root.focus_force()
        self.root.event_generate("<Return>")
        self.root.update()
        entries = [widget for widget in self.widgets if widget.winfo_class() == "Entry"]

        self.assertEqual(1, len(self.received_values))

        self.root.event_generate("<Escape>")
        self.root.update()
        self.assertEqual([""] * 8, [entry.get() for entry in entries])

    def test_small_mousewheel_delta_still_scrolls(self):
        canvases = [widget for widget in self.widgets if widget.winfo_class() == "Canvas"]
        scroll_canvas = next(canvas for canvas in canvases if canvas.cget("yscrollcommand"))
        scroll_canvas.configure(height=100, scrollregion=(0, 0, 100, 1000))
        self.root.update_idletasks()
        scroll_canvas.yview_moveto(0)

        scroll_canvas.event_generate("<MouseWheel>", delta=-1)
        self.root.update_idletasks()

        self.assertGreater(scroll_canvas.yview()[0], 0.0)


if __name__ == "__main__":
    unittest.main()
