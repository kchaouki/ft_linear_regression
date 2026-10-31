#!/usr/bin/env python3

import csv
import json
import math
import sys

DATA_FILE = "data.csv"
THETAS_FILE = "thetas.json"
LEARNING_RATE = 0.001
EPSILON = 1e-12  # stop when the cost improves by less than this

def load_data(path):
    mileages, prices = [], []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                sys.exit(f"Error: '{path}' is empty.")
            for line_num, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) != 2:
                    sys.exit(f"Error: line {line_num}: expected 2 columns, got {len(row)}.")
                try:
                    km, price = float(row[0]), float(row[1])
                except ValueError:
                    sys.exit(f"Error: line {line_num}: non-numeric value {row}.")
                if not (math.isfinite(km) and math.isfinite(price)):
                    sys.exit(f"Error: line {line_num}: value is not finite {row}.")
                mileages.append(km)
                prices.append(price)
    except FileNotFoundError:
        sys.exit(f"Error: '{path}' not found.")
    except PermissionError:
        sys.exit(f"Error: cannot read '{path}' (permission denied).")
    except (UnicodeDecodeError, csv.Error) as e:
        sys.exit(f"Error: '{path}' is not a valid CSV file ({e}).")

    if len(mileages) < 2:
        sys.exit("Error: need at least 2 data rows to train.")
    return mileages, prices


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def cost(x, y, theta0, theta1):
    m = len(x)
    return sum((estimate_price(x[i], theta0, theta1) - y[i]) ** 2 for i in range(m)) / (2 * m)


def gradient_descent(x, y, learning_rate):
    """Find theta0, theta1 minimizing the cost."""
    m = len(x)
    theta0, theta1 = 0.0, 0.0
    prev_cost = cost(x, y, theta0, theta1)

    while True:
        errors = [estimate_price(x[i], theta0, theta1) - y[i] for i in range(m)]
        
        tmp_theta0 = learning_rate * (1 / m) * sum(errors[i] for i in range(m))
        tmp_theta1 = learning_rate * (1 / m) * sum(errors[i] * x[i] for i in range(m))

        # Simultaneous update
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        current_cost = cost(x, y, theta0, theta1)
        if not math.isfinite(current_cost):
            sys.exit("Error: training diverged, lower the learning rate.")
        if abs(prev_cost - current_cost) < EPSILON:
            break
        prev_cost = current_cost

    return theta0, theta1

def main():
    mileages, prices = load_data(DATA_FILE)

    if min(mileages) == max(mileages):
        sys.exit("Error: all mileages are identical, cannot fit a line.")

    # Scale mileage to [0, 1] by dividing by the max: raw mileages (up to
    # ~240000) make the theta1 gradient huge and gradient descent diverges.
    km_max = max(abs(km) for km in mileages)
    x = [km / km_max for km in mileages]

    a, b = gradient_descent(x, prices, LEARNING_RATE)

    # Convert back to raw mileage: price = a + b * (km / km_max)
    theta0 = a
    theta1 = b / km_max

    try:
        with open(THETAS_FILE, "w") as f:
            json.dump({"theta0": theta0, "theta1": theta1}, f, indent=4)
    except OSError as e:
        sys.exit(f"Error: cannot write '{THETAS_FILE}' ({e}).")

    print(f"theta0 = {theta0}")
    print(f"theta1 = {theta1}")
    print(f"Saved to '{THETAS_FILE}'.")

if __name__ == "__main__":
    main()
