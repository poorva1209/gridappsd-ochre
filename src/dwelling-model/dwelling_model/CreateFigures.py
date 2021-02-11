import pandas as pd
import datetime as dt
import string

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

register_matplotlib_converters()
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)


def valid_file(s):
    # removes special characters like ",:$#*" from file names
    valid_chars = "-_.() {}{}".format(string.ascii_letters, string.digits)
    return ''.join(c for c in s if c in valid_chars)


# **** Time-based figures ****


def plot_time_series(df, y_name, ax=None, **kwargs):
    # needs time-based index. Will plot all columns and whole index
    # order = list(df.columns)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    df.plot(ax=ax, **kwargs)

    ax.set_ylabel(y_name)
    ax.set_xlabel('Time')

    return fig


def plot_daily(df_raw, column, plot_average=True, plot_singles=True, plot_min=True, plot_max=True, plot_sd=False,
               **kwargs):
    # sets datetime index to time, by default, plots the average, min, max, and individual days
    # plot_sd: plots a 95% confidence interval, uses average +/- 2 * standard dev.
    df = df_raw.copy()

    assert isinstance(df.index, pd.DatetimeIndex)
    df['Time of Day'] = df.index.time
    df['Date'] = df.index.date
    time_res = df.index[1] - df.index[0]
    # use arbitrary date for plotting
    times = pd.date_range(dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2),
                          freq=time_res, closed='left').to_pydatetime()

    fig, ax = plt.subplots()

    if plot_singles:
        df_singles = pd.pivot(df, 'Time of Day', 'Date', column)
        alpha = kwargs.pop('singles_alpha', 1 / len(df_singles.columns))
        for col in df_singles.columns:
            ax.plot(times, df_singles[col], 'k', alpha=alpha, label=None)

    df_agg = df.groupby('Time of Day')[column].agg(['min', 'max', 'mean', 'std'])
    if plot_max:
        ax.plot(times, df_agg['max'], 'k', label='Maximum')

    if plot_average:
        ax.plot(times, df_agg['mean'], 'b--', label='Average')

    if plot_min:
        ax.plot(times, df_agg['min'], 'k', label='Minimum')

    if plot_sd:
        df_agg['min'] = df_agg['mean'] - 2 * df_agg['std']
        df_agg['max'] = df_agg['mean'] + 2 * df_agg['std']
        alpha = kwargs.pop('std_alpha', 0.4)
        ax.fill_between(times, df_agg['min'], df_agg['max'], alpha=alpha, label='95% C.I.')

    ax.legend()
    ax.set_ylabel(column)
    ax.set_xlabel('Time of Day')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(locator)

    return fig
