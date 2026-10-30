#!/usr/bin/env python3

"""Predict the price of a car for a given mileage using the trained thetas."""

import json
import math
import sys

THETAS_FILE = "thetas.json"

def load_thetas(path):
    try:
        with open(path) as f:
            data = json.load(f)
        theta0, theta1 = float(data["theta0"]), float(data["theta1"])
    except FileNotFoundError:
        return 0.0, 0.0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        sys.exit(f"Error: '{path}' is corrupted, run train.py again.")
    if not (math.isfinite(theta0) and math.isfinite(theta1)):
        sys.exit(f"Error: '{path}' contains invalid values, run train.py again.")
    return theta0, theta1


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def read_mileage():
    try:
        raw = input("Enter a mileage (km): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit("Error: no mileage given.")
    try:
        mileage = float(raw)
    except ValueError:
        sys.exit(f"Error: '{raw}' is not a number.")
    if not math.isfinite(mileage):
        sys.exit("Error: mileage must be a finite number.")
    if mileage < 0:
        sys.exit("Error: mileage cannot be negative.")
    return mileage


def main():
    theta0, theta1 = load_thetas(THETAS_FILE)
    mileage = read_mileage()
    price = estimate_price(mileage, theta0, theta1)

    print(f"Estimated price: {price:.2f}")
    if price < 0:
        print("Warning: this mileage is far outside the training data, "
              "the linear model is not reliable here.")

if __name__ == "__main__":
    main()
