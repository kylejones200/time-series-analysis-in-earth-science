"""Generated from Jupyter notebook: earthpy colorado river

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import os
import warnings

import earthpy as et
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm


def analyze_flood_metrics(discharge_df, precip_df, flood_start, flood_end):
    """Calculate key flood metrics"""
    flood_discharge = subset_time_period(discharge_df, flood_start, flood_end)
    flood_precip = subset_time_period(precip_df, flood_start, flood_end)
    metrics = {
        "total_precip": flood_precip["HPCP"].sum(),
        "max_discharge": flood_discharge["disValue"].max(),
        "mean_discharge": flood_discharge["disValue"].mean(),
        "max_precip_day": flood_precip["HPCP"].idxmax(),
        "max_discharge_day": flood_discharge["disValue"].idxmax(),
    }
    return metrics


def create_animated_plot(
    discharge_df,
    precip_df,
    start_date,
    end_date,
    interval=100,
    save_path="flood_animation.gif",
):
    """
    Create an animated plot showing discharge and precipitation over time
    with progress bar
    """
    discharge_subset = subset_time_period(discharge_df, start_date, end_date)
    precip_subset = subset_time_period(precip_df, start_date, end_date)
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()
    (discharge_line,) = ax1.plot([], [], "b-", label="Discharge", linewidth=2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Discharge Value", color="blue")
    ax2.set_ylabel("Precipitation (mm)", color="red")
    ax1.set_xlim(discharge_subset.index[0], discharge_subset.index[-1])
    ax1.set_ylim(0, discharge_subset["disValue"].max() * 1.1)
    ax2.set_ylim(0, precip_subset["HPCP"].max() * 1.1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.legend(["Discharge"], loc="upper left")
    ax2.legend(["Precipitation"], loc="upper right")
    plt.tight_layout()

    def animate(frame):
        """Animation function"""
        ax2.clear()
        ax2.set_ylabel("Precipitation (mm)", color="red")
        ax2.set_ylim(0, precip_subset["HPCP"].max() * 1.1)
        current_date = discharge_subset.index[frame]
        dates = discharge_subset.index[: frame + 1]
        discharge_values = discharge_subset["disValue"][: frame + 1]
        discharge_line.set_data(dates, discharge_values)
        ax2.bar(
            precip_subset.index[: frame + 1],
            precip_subset["HPCP"][: frame + 1],
            color="red",
            alpha=0.3,
        )
        plt.title(f"2013 Colorado Flood: {current_date.strftime('%Y-%m-%d')}")
        pbar.update(1)
        return (discharge_line,)

    pbar = tqdm(total=len(discharge_subset), desc="Creating animation")
    anim = FuncAnimation(
        fig, animate, frames=len(discharge_subset), interval=interval, blit=False
    )
    writer = PillowWriter(fps=30)
    anim.save(save_path, writer=writer)
    pbar.close()
    return anim


def fit_sarima_model(
    data, train_size=0.8, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)
):
    """Fit SARIMA model and make predictions"""
    train_size = int(len(data) * train_size)
    train = data[:train_size]
    test = data[train_size:]
    model = SARIMAX(train["disValue"], order=order, seasonal_order=seasonal_order)
    results = model.fit()
    predictions = results.predict(start=len(train), end=len(train) + len(test) - 1)
    return (train, test, predictions, results)


def get_noaa_precip_data(station_id, start_date, end_date, token):
    """
    Fetch precipitation data from NOAA API

    Parameters:
    -----------
    station_id: str, NOAA station identifier
    start_date: str, start date in format 'YYYY-MM-DD'
    end_date: str, end date in format 'YYYY-MM-DD'
    token: str, NOAA API token
    """
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": token}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": "PRCP",
        "limit": 1000,
    }
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
    else:
        print(f"Error: {response.status_code}")
        return None


def plot_discharge(df, title="Stream Discharge Over Time"):
    """Plot discharge time series"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["disValue"])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge Value")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    return (fig, ax)


def plot_discharge_and_precip(discharge_df, precip_df, start_date, end_date):
    """Plot combined discharge and precipitation"""
    discharge_subset = subset_time_period(discharge_df, start_date, end_date)
    precip_subset = subset_time_period(precip_df, start_date, end_date)
    fig, ax1 = plt.subplots(figsize=(15, 7))
    color = "tab:blue"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Discharge Value", color=color)
    ax1.plot(
        discharge_subset.index,
        discharge_subset["disValue"],
        color=color,
        label="Discharge",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Precipitation (mm)", color=color)
    ax2.bar(
        precip_subset.index,
        precip_subset["HPCP"],
        color=color,
        alpha=0.3,
        label="Precipitation",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(bottom=0)
    plt.title("Stream Discharge and Precipitation During 2013 Colorado Flood")
    plt.xticks(rotation=45)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    return (fig, (ax1, ax2))


def plot_sarima_results(
    train, test, predictions, confidence_intervals=None, title="SARIMA Forecast"
):
    """Plot SARIMA results with confidence intervals"""
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(train.index, train["disValue"], label="Training Data", color="blue")
    ax.plot(test.index, test["disValue"], label="Actual Test Data", color="green")
    ax.plot(test.index, predictions, label="SARIMA Forecast", color="red")
    if confidence_intervals is not None:
        lower_ci = np.maximum(confidence_intervals["lower"], 0)
        ax.fill_between(
            test.index,
            lower_ci,
            confidence_intervals["upper"],
            color="red",
            alpha=0.1,
            label="95% Confidence Interval",
        )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge Value")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.legend()
    return (fig, ax)


def process_discharge_data(filepath):
    """Read and process discharge data"""
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    return df


def process_precip_data(filepath):
    """Read and process precipitation data"""
    df = pd.read_csv(filepath)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.set_index("DATE", inplace=True)
    return df


def resample_weekly(df):
    """Resample data to weekly values"""
    return df.resample("W").max()


def save_plot(fig, filename, dpi=300):
    """Save plot to file"""
    if not os.path.exists("plots"):
        os.makedirs("plots")
    fig.savefig(os.path.join("plots", filename), bbox_inches="tight", dpi=dpi)


def subset_time_period(df, start_date, end_date):
    """Subset data for specific time period"""
    return df.loc[start_date:end_date]


def download_example_data() -> None:
    data = et.data.get_data("cold-springs-fire")

    print("Data directory contents:")

    print(os.listdir(data))

    possible_paths = [
        os.path.join(data, "modis_lst_example.csv"),
        os.path.join(data, "outputs", "modis_lst_example.csv"),
        os.path.join(data, "data", "modis_lst_example.csv"),
    ]

    lst_path = None

    for path in possible_paths:
        if os.path.exists(path):
            lst_path = path
            break

    if lst_path is None:
        print(
            "Could not find the MODIS LST data file. Please check the file structure."
        )
    else:
        df = pd.read_csv(lst_path)
        df["date"] = pd.to_datetime(df["date"])
        print("\nDataset head:")
        print(df.head())


def notebook_step_002() -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(df["date"], df["temperature"], label="Land Surface Temperature")

    plt.title("Land Surface Temperature Over Time")

    plt.xlabel("Date")

    plt.ylabel("Temperature (°C)")

    plt.legend()

    plt.grid(True)

    plt.savefig("land_surface_temperature_over_time.png")

    plt.show()


def set_the_date_column_as_the_index_for_decompositi() -> None:
    df.set_index("date", inplace=True)

    result = seasonal_decompose(df["temperature"], model="additive", period=365)

    result.plot()

    plt.savefig("seasonal_decomposition.png")

    plt.show()

    rolling_mean = df["temperature"].rolling(window=30).mean()

    rolling_std = df["temperature"].rolling(window=30).std()

    anomalies = df[
        (df["temperature"] > rolling_mean + 2 * rolling_std)
        | (df["temperature"] < rolling_mean - 2 * rolling_std)
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(df.index, df["temperature"], label="Temperature")

    plt.scatter(
        anomalies.index, anomalies["temperature"], color="red", label="Anomalies"
    )

    plt.title("Temperature Anomalies")

    plt.xlabel("Date")

    plt.ylabel("Temperature (°C)")

    plt.legend()

    plt.grid(True)

    plt.savefig("temperature_anomalies.png")

    plt.show()

    model = ARIMA(df["temperature"], order=(5, 1, 0))

    fit = model.fit()

    forecast = fit.forecast(steps=365)

    plt.figure(figsize=(10, 6))

    plt.plot(df.index, df["temperature"], label="Historical Temperature")

    plt.plot(
        pd.date_range(df.index[-1], periods=365, freq="D"), forecast, label="Forecast"
    )

    plt.title("Temperature Forecast")

    plt.xlabel("Date")

    plt.ylabel("Temperature (°C)")

    plt.legend()

    plt.grid(True)

    plt.savefig("temperature_forecast.png")

    plt.show()


def earth_data_science_tutorial_blog() -> None:
    warnings.filterwarnings("ignore")

    register_matplotlib_converters()

    sns.set(font_scale=1.5, style="whitegrid")

    data = et.data.get_data("colorado-flood")

    os.chdir(os.path.join(et.io.HOME, "earth-analytics", "data"))

    stream_discharge_path = os.path.join(
        "colorado-flood", "discharge", "06730200-discharge-daily-1986-2013.csv"
    )

    if __name__ == "__main__":
        print("Loading and processing data...")
        discharge_df = process_discharge_data(stream_discharge_path)
        fig1, ax1 = plot_discharge(discharge_df)
        save_plot(fig1, "full_timeseries.png")
        flood_period = subset_time_period(discharge_df, "2013-08-01", "2013-10-31")
        fig2, ax2 = plot_discharge(
            flood_period, "Stream Discharge During 2013 Colorado Flood"
        )
        save_plot(fig2, "flood_period.png")
        weekly_data = resample_weekly(flood_period)
        fig3, ax3 = plot_discharge(weekly_data, "Weekly Maximum Stream Discharge")
        save_plot(fig3, "weekly_discharge.png")
        print("Fitting SARIMA model...")
        train, test, predictions, model_results = fit_sarima_model(
            discharge_df, train_size=0.8, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)
        )
        forecast = model_results.get_forecast(len(test))
        conf_int = forecast.conf_int()
        confidence_intervals = {
            "lower": conf_int.iloc[:, 0],
            "upper": conf_int.iloc[:, 1],
        }
        rmse = np.sqrt(mean_squared_error(test["disValue"], predictions))
        print(f"RMSE: {rmse:.2f}")
        fig4, ax4 = plot_sarima_results(
            train,
            test,
            predictions,
            confidence_intervals,
            "SARIMA Forecast vs Actual Discharge",
        )
        save_plot(fig4, "sarima_forecast.png")
        last_year = discharge_df.last("365D")
        last_year_train = train[train.index >= last_year.index[0]]
        last_year_test = test[test.index >= last_year.index[0]]
        last_year_pred = predictions[predictions.index >= last_year.index[0]]
        last_year_ci = {
            "lower": confidence_intervals["lower"][
                confidence_intervals["lower"].index >= last_year.index[0]
            ],
            "upper": confidence_intervals["upper"][
                confidence_intervals["upper"].index >= last_year.index[0]
            ],
        }
        fig5, ax5 = plot_sarima_results(
            last_year_train,
            last_year_test,
            last_year_pred,
            last_year_ci,
            "SARIMA Forecast vs Actual Discharge (Last Year)",
        )
        save_plot(fig5, "sarima_forecast_zoomed.png")
        plt.show()


def add_to_imports() -> None:
    precip_path = os.path.join(
        "colorado-flood", "precipitation", "805325-precip-daily-2003-2013.csv"
    )

    if __name__ == "__main__":
        print("Loading and processing discharge data...")
        discharge_df = process_discharge_data(stream_discharge_path)
        print("Loading and processing precipitation data...")
        precip_df = process_precip_data(precip_path)
        print("Creating combined discharge and precipitation plot...")
        fig6, axes = plot_discharge_and_precip(
            discharge_df, precip_df, "2013-09-01", "2013-09-30"
        )
        save_plot(fig6, "discharge_and_precip_flood.png")
        flood_precip = precip_df.loc["2013-09-01":"2013-09-30", "HPCP"].sum()
        print(f"Total precipitation during flood period: {flood_precip:.1f} mm")


def earth_data_science_tutorial_blog_2() -> None:
    warnings.filterwarnings("ignore")

    register_matplotlib_converters()

    sns.set(font_scale=1.5, style="whitegrid")

    data = et.data.get_data("colorado-flood")

    os.chdir(os.path.join(et.io.HOME, "earth-analytics", "data"))

    stream_discharge_path = os.path.join(
        "colorado-flood", "discharge", "06730200-discharge-daily-1986-2013.csv"
    )

    precip_path = os.path.join(
        "colorado-flood", "precipitation", "805325-precip-daily-2003-2013.csv"
    )

    if __name__ == "__main__":
        print("Loading and processing data...")
        discharge_df = process_discharge_data(stream_discharge_path)
        precip_df = process_precip_data(precip_path)
        flood_metrics = analyze_flood_metrics(
            discharge_df, precip_df, "2013-09-01", "2013-09-30"
        )
        print("\nFlood Metrics:")
        for key, value in flood_metrics.items():
            print(f"{key}: {value}")
        print("\nGenerating plots...")
        fig1, ax1 = plot_discharge(discharge_df)
        save_plot(fig1, "full_timeseries.png")
        fig2, axes2 = plot_discharge_and_precip(
            discharge_df, precip_df, "2013-09-01", "2013-09-30"
        )
        save_plot(fig2, "flood_period_combined.png")
        print("\nFitting SARIMA model...")
        train, test, predictions, model_results = fit_sarima_model(
            discharge_df, train_size=0.8, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)
        )
        forecast = model_results.get_forecast(len(test))
        conf_int = forecast.conf_int()
        confidence_intervals = {
            "lower": conf_int.iloc[:, 0],
            "upper": conf_int.iloc[:, 1],
        }
        rmse = np.sqrt(mean_squared_error(test["disValue"], predictions))
        print(f"RMSE: {rmse:.2f}")
        fig3, ax3 = plot_sarima_results(
            train,
            test,
            predictions,
            confidence_intervals,
            "SARIMA Forecast vs Actual Discharge",
        )
        save_plot(fig3, "sarima_forecast.png")
        plt.show()


def add_these_imports_at_the_top() -> None:
    if __name__ == "__main__":
        print("\nCreating animation...")
        anim = create_animated_plot(
            discharge_df,
            precip_df,
            "2013-09-01",
            "2013-09-30",
            interval=50,
            save_path="flood_animation.gif",
        )
        plt.show()


def add_these_imports_at_the_top_2() -> None:
    if __name__ == "__main__":
        print("\nCreating animation...")
        anim = create_animated_plot(
            discharge_df,
            precip_df,
            "2013-09-01",
            "2013-09-30",
            interval=50,
            save_path="flood_animation.gif",
        )
        plt.show()


def main() -> None:
    download_example_data()
    notebook_step_002()
    set_the_date_column_as_the_index_for_decompositi()
    earth_data_science_tutorial_blog()
    add_to_imports()
    earth_data_science_tutorial_blog_2()
    add_these_imports_at_the_top()
    add_these_imports_at_the_top_2()


if __name__ == "__main__":
    main()
