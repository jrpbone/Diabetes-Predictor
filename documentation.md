# Diabetes Prediction Runtime Model

## Overview

This codebase implements a lightweight runtime "model" for predicting diabetes risk using basic statistical heuristics derived from a CSV dataset.

The program can run in two modes:

1. **GUI Mode** – If `diabetes_gui.window` is available.
2. **CLI Mode** – Falls back to terminal input if the GUI module is not found.

Unlike a traditional machine learning model, this system does **not** train on labeled outcomes. Instead, it derives feature weights from dataset statistics (ranges and means) and applies a linear scoring rule.

---

# High-Level Workflow

The complete data pipeline looks like this:

```
CSV File
   ↓
read_csv()
   ↓
Raw Numeric Rows (8 features each)
   ↓
analyze()
   ↓
Dataset Statistics (means, mins, maxs, spreads, stds)
   ↓
derive_weights()
   ↓
Model Parameters (weights + bias + threshold)
   ↓
predict() / predict_with_context()
   ↓
Binary Prediction (+ optional likelihood %)
```

---

# Module Breakdown

## 1. Optional GUI Import

```python
try:
    from diabetes_gui import window
except Exception:
    window = None
```

### Purpose
Allows the program to run with a GUI if available, otherwise default to CLI mode.

### Behavior
- If import succeeds → GUI mode available.
- If import fails → `window` is set to `None` and CLI mode is used.

---

## 2. Configuration Constants

```python
FEATURE_COUNT = 8
DEFAULT_CSV_PATH = "diabetes.csv"
```

- The model expects exactly **8 numeric inputs per row**.
- The default dataset file is `diabetes.csv`.

`FEATURE_LABELS` defines human-readable names used in GUI and CLI prompts.

---

## 3. Parsing Helper

### `try_float(text)`

**Input:** string  
**Output:** float or `None`

Safely converts a string into a float. If conversion fails, returns `None` instead of raising an exception.

This prevents crashes while reading CSV data.

---

## 4. Math Helpers

### `approximate_exp(value)`

Approximates:

```
e^x = 1 + x + x²/2! + x³/3! + ...
```

- Clamps input to [-60, 60] to avoid overflow.
- Uses 40 Taylor series terms.

### `sigs(value)`

Implements the logistic sigmoid function:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Output is between 0 and 1.

Used only for generating a smooth "likelihood percentage".

---

# Core Data Processing

## 5. `read_csv(path)`

### Input
Path to CSV file.

### Output
```
List[List[float]]
```

Each row contains exactly 8 numeric feature values.

### Process
1. Read file line by line.
2. Split each line by comma.
3. Convert values using `try_float()`.
4. Collect the first 8 numeric values.
5. Only keep rows that contain exactly 8 valid numeric entries.

### Transformation Example

Raw CSV line:
```
6,148,72,35,0,33.6,0.627,50,1
```

Becomes:
```
[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]
```

---

## 6. `analyze(data)`

### Input
```
List[List[float]]
```

### Output
Dictionary containing:
- count
- mins
- maxs
- means
- spreads (max - min)
- stds (population standard deviation)

### Process

First Pass:
- Compute sums
- Compute per-feature min/max

Second Pass:
- Compute variance sums
- Compute standard deviations

### Purpose
Transforms raw rows into statistical summaries of the dataset.

---

## 7. `derive_weights(stats)`

### Input
Statistics dictionary

### Output
```
(m_list, b, threshold)
```

Where:
- `m_list` = feature weights
- `b` = bias
- `threshold` = 0.5

### How Weights Are Derived

1. Compute inverse feature ranges:
   ```
   inv[j] = 1 / spread[j]
   ```
2. Normalize so weights sum to 1.

Features with smaller spread receive larger weight.

### Bias Calculation

Compute average dataset score:

```
avg_score = sum(weight[j] * mean[j])
```

Set bias so dataset mean lies exactly on decision boundary:

```
b = threshold - avg_score
```

---

# Prediction Stage

## 8. `predict(m_list, b, threshold, x_instance)`

### Input
- Model parameters
- One sample with 8 features

### Computation

Linear score:

```
score = b + Σ(weight[j] * x[j])
```

### Output
- `1` if score >= threshold
- `0` otherwise

This is a heuristic linear classifier.

---

## 9. `predict_with_context(...)`

Extends `predict()` by adding a smooth confidence-like percentage.

### Additional Step

```
likelihood = sigmoid(score - threshold) * 100
```

Produces output:

```
(label, likelihood_percent)
```

---

# Runtime Model Builder

## 10. `build_runtime_model(path)`

Pipeline function:

```
read_csv() → analyze() → derive_weights()
```

Returns model parameters or `None` if dataset cannot be loaded.

---

# Application Entry Point

## `main()`

### Steps

1. Build model from CSV.
2. Create GUI wrapper function.
3. If GUI exists → launch GUI.
4. Otherwise → prompt user for 8 inputs via CLI.
5. Run prediction and print result.

---

# Known Issue

In CLI mode:

```
value = _try_float(raw_value.strip())
```

Should be:

```
value = try_float(raw_value.strip())
```

Otherwise a `NameError` will occur.

---

# Important Notes

- This is **not** a trained machine learning model.
- No outcome labels are used.
- Weights are derived purely from feature ranges.
- It behaves like a normalized linear scoring heuristic.

---

# Summary

This system:

- Loads dataset
- Extracts numeric rows
- Computes descriptive statistics
- Derives feature weights from ranges
- Applies a linear scoring rule
- Optionally produces a sigmoid-based likelihood percentage

It demonstrates a full data pipeline from CSV ingestion to runtime prediction while remaining dependency-light and GUI-optional.

