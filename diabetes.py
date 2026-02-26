# Try catch for import options
try:
    from diabetes_gui import window
except Exception:
    window = None

#Global vars
FEATURE_COUNT = 8
DEFAULT_CSV_PATH = "diabetes.csv"
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

# helper function for float
def _try_float(text):
    try:
        return float(text)
    except Exception:
        return None


def _approximate_exp(value):
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


def _sigmoid(value):
    return 1.0 / (1.0 + _approximate_exp(-value))

# read csv file
def read_csv(path=DEFAULT_CSV_PATH):                            #path leads to the csv file global var
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

        parts = line.split(",")
        row = []

        i = 0
        while i < len(parts):
            value = _try_float(parts[i].strip())
            if value is not None:
                row.append(value)
                if len(row) == FEATURE_COUNT:
                    break
            i += 1

        # Keep only rows with at least 8 numeric values.
        if len(row) == FEATURE_COUNT:
            data.append(row)

    return data


def analyze(data):                                              #data from read_csv goes through here
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

    j = 0
    while j < FEATURE_COUNT:
        stats["mins"][j] = data[0][j]
        stats["maxs"][j] = data[0][j]
        j += 1

    i = 0
    sums = [0.0] * FEATURE_COUNT
    while i < n:
        row = data[i]
        j = 0
        while j < FEATURE_COUNT:
            x = row[j]
            sums[j] += x
            if x < stats["mins"][j]:
                stats["mins"][j] = x
            if x > stats["maxs"][j]:
                stats["maxs"][j] = x
            j += 1
        i += 1

    j = 0
    while j < FEATURE_COUNT:
        stats["means"][j] = sums[j] / n
        stats["spreads"][j] = stats["maxs"][j] - stats["mins"][j]
        j += 1

    # Optional std (manual)
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

    j = 0
    while j < FEATURE_COUNT:
        stats["stds"][j] = (var_sums[j] / n) ** 0.5
        j += 1

    return stats


def derive_weights(stats):
    spreads = stats["spreads"]
    means = stats["means"]

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

    m_list = [0.0] * FEATURE_COUNT
    if total_inv > 0:
        j = 0
        while j < FEATURE_COUNT:
            m_list[j] = inv[j] / total_inv
            j += 1

    threshold = 0.5

    avg_score = 0.0
    j = 0
    while j < FEATURE_COUNT:
        avg_score += m_list[j] * means[j]
        j += 1

    # Makes an average row score near threshold.
    b = threshold - avg_score

    return m_list, b, threshold


def predict(m_list, b, threshold, x_instance):
    score = b
    j = 0
    while j < FEATURE_COUNT:
        score += m_list[j] * x_instance[j]
        j += 1
    return 1 if score >= threshold else 0


def predict_with_context(m_list, b, threshold, x_instance):
    score = b
    j = 0
    while j < FEATURE_COUNT:
        score += m_list[j] * x_instance[j]
        j += 1

    label = 1 if score >= threshold else 0
    # Convert distance from threshold to a 0-100 likelihood score.
    likelihood_percent = _sigmoid(score - threshold) * 100.0

    return label, likelihood_percent


def build_runtime_model(path=DEFAULT_CSV_PATH):
    data = read_csv(path)
    if len(data) == 0:
        return None

    stats = analyze(data)
    return derive_weights(stats)


def main():
    model = build_runtime_model()
    if model is None:
        return

    m_list, b, threshold = model

    def gui_predictor(x_instance):
        return predict_with_context(m_list, b, threshold, x_instance)

    if window is not None:
        window(FEATURE_LABELS, gui_predictor)
        return

    x_instance = []
    i = 0
    while i < FEATURE_COUNT:
        try:
            raw_value = input(FEATURE_LABELS[i] + ": ")
        except Exception:
            return

        value = _try_float(raw_value.strip())
        if value is None:
            return

        x_instance.append(value)
        i += 1

    y = predict(m_list, b, threshold, x_instance)
    print(y)


if __name__ == "__main__":
    main()
