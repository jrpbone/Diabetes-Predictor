# No imports used.

FEATURE_COUNT = 8
DEFAULT_CSV_PATH = "diabetes.csv"


def _try_float(text):
    try:
        return float(text)
    except Exception:
        return None


def read_csv(path=DEFAULT_CSV_PATH):
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


def analyze(data):
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


def _parse_instance(line):
    line = line.strip()
    if line == "":
        return None

    line = line.replace(",", " ")
    parts = line.split()

    values = []
    i = 0
    while i < len(parts):
        v = _try_float(parts[i])
        if v is not None:
            values.append(v)
            if len(values) == FEATURE_COUNT:
                break
        i += 1

    if len(values) != FEATURE_COUNT:
        return None
    return values


def main():
    # Input line: 8 feature values (space or comma separated)
    try:
        instance_line = input()
    except Exception:
        return

    data = read_csv()
    if len(data) == 0:
        return

    x_instance = _parse_instance(instance_line)
    if x_instance is None:
        return

    stats = analyze(data)
    m_list, b, threshold = derive_weights(stats)
    y = predict(m_list, b, threshold, x_instance)

    # Final output line must be only 1 or 0.
    print(y)


if __name__ == "__main__":
    main()
