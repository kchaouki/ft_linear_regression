#!/usr/bin/env python3

"""Measure how well the trained model fits the dataset."""

import math

from predict import THETAS_FILE, estimate_price, load_thetas
from train import DATA_FILE, load_data


def main():
    mileages, prices = load_data(DATA_FILE)
    theta0, theta1 = load_thetas(THETAS_FILE)
    m = len(mileages)

    if theta0 == 0 and theta1 == 0:
        print("Warning: model not trained yet, run train.py first.\n")

    errors = [estimate_price(mileages[i], theta0, theta1) - prices[i] for i in range(m)]
    mean_price = sum(prices) / m

    # Mean Absolute Error: average distance between prediction and real price.
    mae = sum(abs(e) for e in errors) / m
    
    # Mean Squared Error: same, but big errors count more.
    mse = sum(e ** 2 for e in errors) / m
    
    # Root Mean Squared Error: MSE back in price units.
    rmse = math.sqrt(mse)
    
    # R²: share of the price variation explained by the mileage (1 = perfect).
    ss_res = sum(e ** 2 for e in errors)
    ss_tot = sum((p - mean_price) ** 2 for p in prices)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    print(f"Number of cars : {m}")
    print(f"MAE            : {mae:.2f}")
    print(f"MSE            : {mse:.2f}")
    print(f"RMSE           : {rmse:.2f}")
    print(f"R²             : {r2:.4f}  ({r2 * 100:.2f}% of the price variation explained)")


if __name__ == "__main__":
    main()
