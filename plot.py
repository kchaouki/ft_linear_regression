#!/usr/bin/env python3

import matplotlib.pyplot as plt

from predict import THETAS_FILE, estimate_price, load_thetas
from train import DATA_FILE, load_data

def main():
    mileages, prices = load_data(DATA_FILE)
    theta0, theta1 = load_thetas(THETAS_FILE)

    plt.scatter(mileages, prices, color="tab:blue", label="Data")

    if theta0 == 0 and theta1 == 0:
        print("Model not trained yet, run train.py to see the regression line.")
    else:
        # A line only needs its two end points.
        line_x = [min(mileages), max(mileages)]
        line_y = [estimate_price(km, theta0, theta1) for km in line_x]
        plt.plot(line_x, line_y, color="tab:red",
                 label=f"price = {theta0:.2f} + ({theta1:.6f} * km)")

    plt.title("Car price vs mileage")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
