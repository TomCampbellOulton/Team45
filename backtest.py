import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Setup
REAL_FILE = "Newest Results/Seer.csv"
REAL_COL = "close"

PRED_FILES = [
    (f"Newest Results/{file}", "close") for file in os.listdir("Newest Results/") if file.endswith(".csv") and file != "Seer.csv"
]

CAPITAL = 10_000#_000_000
# % of sale that the broker takes
TX_COST_PCT = 0.1
# The % rise required to go long (how much does it need to rise for us to buy?)
BUY_THRESHOLD = 1.5
# The % drop needed to go short (how much does it need for us to sell / go short?)
SELL_THRESHOLD = 1.5
# Are we allowing short?
ALLOW_SHORT = True

# Set random seeds for reproducibility
RANDOM_SEED = 42
RANDOM_TRIALS = 20


# Load the csv data to numpy array using pandas
def load_csv(path, col):
    df = pd.read_csv(path)
    df.set_index(df.columns[0], inplace=True)

    # Forward fill the missing values ******************************************************************IMPORTANT DESIGN QUESTION FOR GROUP ****************************************
    df = df.ffill()

    return df[col].to_numpy(dtype=float), df.index


# The actual backtest
def backtest(positions, prices):
    pv, prev = CAPITAL, 0
    curve = [pv]
    for i in range(len(prices) - 1):
        if positions[i] != prev:
            pv *= 1 - TX_COST_PCT / 100
        pv *= 1 + positions[i] * (prices[i + 1] - prices[i]) / prices[i]
        curve.append(pv)
        prev = positions[i]
    return np.array(curve)

# Baseline Strategies

# Just buy it and hold until the end of period
def buy_and_hold(prices):
    return np.ones(len(prices))

# Listens to the seer and bets correctly (best possible theoretical yield)
def perfect_seer(prices):
    # +1 for price going up, -1 for price going down
    pos = np.sign(np.diff(prices))
    # The closing position on the very last day :o
    return np.append(pos, 0)

# Listens to the seer, bets wrong everytime...
def worst_case(prices):
    return -perfect_seer(prices)

# Random decisions each day
def random_strat(prices):
    choices = [-1, 0, 1] if ALLOW_SHORT else [0, 1]
    trials = []
    for t in range(RANDOM_TRIALS):
        rng = np.random.default_rng(RANDOM_SEED + t)
        trials.append(rng.choice(choices, size=len(prices)).astype(float))
    return np.mean(trials, axis=0)

# The strategy the model will use - strat is buy (go long) if the current price
def model_strat(prices, predictions):
    # Get % diff between predicted and real prices - if + then predicted is higher than current
    signal = (predictions - prices) / prices * 100

    # Create array to store our strat - [-1, 1, 0] means go short go long then stay neutral
    pos = np.zeros(len(prices))
    # If the signal is sufficiently strong (we're happy it's a good strong signal) then buy (go long)
    pos[signal >= BUY_THRESHOLD] =  1
    # If the signal is too weak (we're fairly sure it's going to drop in price) then go short if allowing it, otherwise stay neutral
    pos[signal <= -SELL_THRESHOLD] = -1 if ALLOW_SHORT else 0
    return pos


# Main

def main(show=False, save=True):

    prices, dates = load_csv(REAL_FILE, REAL_COL)

    strategies = {
        "Buy and Hold": buy_and_hold(prices),
        "Best Case Scenario": perfect_seer(prices),
        "Worst Case Scenario": worst_case(prices),
        "Random Choices": random_strat(prices),
    }



    # Get the values from each csv
    for path, col in PRED_FILES:
        print(path)
        predictions, prediction_dates = load_csv(path, col)

        # Convert predictions to df
        predicted_series = pd.Series(predictions, index=pd.to_datetime(prediction_dates))
        # Get rid of any duplicated predictions by keeping the last prediction for each date
        predicted_series = predicted_series.groupby(predicted_series.index).last()
        
        # Re allign the dates to make sure they match
        aligned_dates = pd.to_datetime(dates)
        aligned_predictions = predicted_series.reindex(aligned_dates, method='ffill')

        # Get a numpy array for the results
        aligned_predictions = aligned_predictions.to_numpy()

        # Record this strat
        name = path.replace(".csv", "")
        strategies[name] = model_strat(prices, aligned_predictions)

    # Execute the backtest
    results = {name: backtest(pos, prices) for name, pos in strategies.items()}


    # Print summary
    print(f"\n{'Strategy':<25} {'Return':>9} {'Final Value':>12}")

    for name, curve in results.items():
        ret = (curve[-1] / curve[0] - 1) * 100
        #print(f"{name:<25} {ret:>+8.1f}%  ${curve[-1]:>10,.2f}")
        print(f"{ret:>+8.1f}%  ${curve[-1]:>10,.2f} {name:<25}")


    # Display our results
    for name, curve in results.items():
        plt.plot(aligned_dates, curve, label=name)

    plt.title("Equity Curves")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("Newest Results/Testing Results/Backtest With Best.png")

    if show:
        plt.show()

    # Now display our results again but without the best (overshadowing)
    for name, curve in results.items():
        if name != "Best Case Scenario":
            plt.plot(aligned_dates, curve, label=name)

    plt.title("Equity Curves")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("Newest Results/Testing Results/Backtest Without Best")

    if show:
        plt.show()

if __name__ == "__main__":
    main(show=True, save=True)