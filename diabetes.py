#Try/except for optional GUI import
#This codebase can run either:
# 1. with a GUI (diabetes_gui.window), or
# 2. as a command-line program.
#If the GUI module isn't available, it falls back to CLI input.
try:
    from diabetes_gui import window
except Exception:
    window = None


#The model expects EXACTLY 8 inputs per row
FEATURE_COUNT = 8

#Default file to read training-like data from at runtime
DEFAULT_CSV_PATH = "diabetes.csv"

#Independent variables
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


#Parsing
def try_float(text):
    """
    Convert a string into a float.

    Why this matters:
    1. The CSV may contain non-numeric fields.
    2. This function safely returns None instead of crashing.
    """
    try:
        return float(text)
    except Exception:
        return None

#Math helpers (for percentage)
def approximate_exp(value):
    """
    Key details:
    - It clamps input to [-60, 60] to avoid overflow/huge numbers.
    - It sums 40 terms of the series:
        e^x = 1 + x + x^2/2! + x^3/3! + ... etc

    Why it exists:
    - Used only for computing a "likelihood %" in predict_with_context().
    """
    if value > 60:
        value = 60
    elif value < -60:
        value = -60

    total = 1.0
    term = 1.0
    n = 1
    while n <= 40:
        term = term * value / n
        total += term
        n += 1

    return total

def sigs(value):
    """
    Logistic sigmoid function:
        sigmoid(z) = 1 / (1 + eulers^-z)

    It's used to convert "distance from decision threshold"
    """
    return 1.0 / (1.0 + approximate_exp(-value))


#Step 1: Read CSV into numeric feature rows

def read_csv(path=DEFAULT_CSV_PATH):
    """
    Load the CSV file and extract numeric rows per intependent variable

    What this function does:
    1. Reads every line of the file.
    2. Splits each line by commas.
    3. Parses each cell as float.
    4. Collects the first 8 numeric values in that line as a "row".
    5. Keeps the row ONLY if it got exactly 8 numeric values.
    """

    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return []

    for line in lines:
        line = line.strip()
        if line == "":
            continue

        parts = line.split(",")     #splitter per value
        row = []

        i = 0
        while i < len(parts):
            value = try_float(parts[i].strip())
            if value is not None:
                row.append(value)

                if len(row) == FEATURE_COUNT:         #Stop as soon as we collected 8 feature values.
                    break
            i += 1

        if len(row) == FEATURE_COUNT:                 #Keep only rows with exactly 8 numeric values.
            data.append(row)

    return data

#Analyze dataset to compute summary statistics
def analyze(data):
    """
    Compute basic descriptive statistics across the dataset. Data are the list of rows, each row is a list[float] of length 8

    Outputs are:
    count: number of rows
    mins[j], maxs[j]: per-feature min/max
    means[j]: per-feature mean
    spreads[j] = maxs[j] - mins[j]: per-feature range
    stds[j]: per-feature standard deviation (population std, divides by n)
    spreads and means feed into derive_weights(), which creates the model
    """
    stats = {
        "count": 0,
        "mins": [0.0] * FEATURE_COUNT,
        "maxs": [0.0] * FEATURE_COUNT,
        "means": [0.0] * FEATURE_COUNT,
        "spreads": [0.0] * FEATURE_COUNT,
        "stds": [0.0] * FEATURE_COUNT,
    }

    n = len(data)
    stats["count"] = n
    if n == 0:
        return stats

    #Initialize mins/maxs with the first row values.
    j = 0
    while j < FEATURE_COUNT:
        stats["mins"][j] = data[0][j]
        stats["maxs"][j] = data[0][j]
        j += 1

    #First pass: compute sums, mins, maxs
    i = 0
    sums = [0.0] * FEATURE_COUNT
    while i < n:
        row = data[i]
        j = 0
        while j < FEATURE_COUNT:
            x = row[j]
            sums[j] += x

            # Track min/max for each feature column
            if x < stats["mins"][j]:
                stats["mins"][j] = x
            if x > stats["maxs"][j]:
                stats["maxs"][j] = x
            j += 1
        i += 1

    #Compute means and spreads (range) per feature
    j = 0
    while j < FEATURE_COUNT:
        stats["means"][j] = sums[j] / n
        stats["spreads"][j] = stats["maxs"][j] - stats["mins"][j]
        j += 1

    #Second pass: compute variance sums (for std dev)
    #Note: uses population variance (divide by n), not sample (n-1).
    var_sums = [0.0] * FEATURE_COUNT
    i = 0
    while i < n:
        row = data[i]
        j = 0
        while j < FEATURE_COUNT:
            d = row[j] - stats["means"][j]
            var_sums[j] += d * d
            j += 1
        i += 1

    # Finalize standard deviations
    j = 0
    while j < FEATURE_COUNT:
        stats["stds"][j] = (var_sums[j] / n) ** 0.5
        j += 1

    return stats

#Train a very simple runtime model (weights + bias + threshold)

def derive_weights(stats):
    """
    Derive a set of weights (m_list), a bias term (b), and a decision threshold.

    The "average" row from the dataset sits right at the decision boundary.
    Rows scoring above that are predicted 1, below that predicted 0.
    """
    spreads = stats["spreads"]
    means = stats["means"]

    # Compute inverse spreads (1/range).
    inv = [0.0] * FEATURE_COUNT
    total_inv = 0.0

    j = 0
    while j < FEATURE_COUNT:
        s = spreads[j]
        if s > 0:
            inv[j] = 1.0 / s
        else:
            inv[j] = 0.0
        total_inv += inv[j]
        j += 1

    #Normalize so weights sum to 1 (convex combination of features).
    m_list = [0.0] * FEATURE_COUNT
    if total_inv > 0:
        j = 0
        while j < FEATURE_COUNT:
            m_list[j] = inv[j] / total_inv
            j += 1

    # Fixed cutoff used by predict():
    threshold = 0.5

    #Compute average (mean-row) score under these weights.
    avg_score = 0.0
    j = 0
    while j < FEATURE_COUNT:
        avg_score += m_list[j] * means[j]
        j += 1

    #Pick bias so that the dataset mean maps to the threshold.
    b = threshold - avg_score

    return m_list, b, threshold

#Predict (linear rule)
def predict(m_list, b, threshold, x_instance):
    """
    Produce a binary prediction for one instance.

    Because weights/bias are not learned from labeled outcomes,
    this is a heuristic classifier, not a trained diabetes model.
    """
    score = b
    j = 0
    while j < FEATURE_COUNT:
        score += m_list[j] * x_instance[j]
        j += 1
    return 1 if score >= threshold else 0


def predict_with_context(m_list, b, threshold, x_instance):
    """
    Interpretation:
    - It is a smooth confidence-like indicator:
        * Far above threshold => approaches 100%
        * Near threshold => around 50%
        * Far below threshold => approaches 0%
    """
    score = b
    j = 0
    while j < FEATURE_COUNT:
        score += m_list[j] * x_instance[j]
        j += 1

    label = 1 if score >= threshold else 0

    # Convert distance from threshold into a 0-100 "likelihood".
    likelihood_percent = sigs(score - threshold) * 100.0

    return label, likelihood_percent


#Build model at runtime from CSV
def build_runtime_model(path=DEFAULT_CSV_PATH):

    data = read_csv(path)
    if len(data) == 0:
        return None

    stats = analyze(data)
    return derive_weights(stats)

def main():
    # Build the model parameters from the CSV at startup.
    model = build_runtime_model()
    if model is None:
        return

    m_list, b, threshold = model

    # Wrap the prediction function so the GUI can call it.
    def gui_predictor(x_instance):
        return predict_with_context(m_list, b, threshold, x_instance)

    # If GUI is available, run GUI and exit.
    if window is not None:
        window(FEATURE_LABELS, gui_predictor)
        return

    # Otherwise, collect feature inputs from the user in the terminal.
    x_instance = []
    i = 0
    while i < FEATURE_COUNT:
        try:
            raw_value = input(FEATURE_LABELS[i] + ": ")
        except Exception:
            return

        value = try_float(raw_value.strip())
        if value is None:
            # Invalid input => quit (could be improved with re-prompt).
            return

        x_instance.append(value)
        i += 1

    # Run prediction and print the binary result.
    y = predict(m_list, b, threshold, x_instance)
    print(y)


if __name__ == "__main__":
    main()