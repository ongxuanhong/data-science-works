# Given:
# - List of creative ids
# Output:
# - CSV file for each creative with (Moving average)
# Params: --start_hour="2017050100" --end_hour="2017060100" --widget_ids="knxad_knx2991_201604204777" --adunit_ids="ea845d619a665959b6a6d7415457479a55cd4149"
import argparse
import calendar
import datetime
import time

import pandas as pd
from pymongo import MongoClient


def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"


def cal_click_through(interactions):
    """
    Get click-through count from interactions
    """
    click_through = 0
    for i in range(0, len(interactions)):
        interact = interactions[i]
        if interact["clickThrough"]:
            click_through += interact["count"]

    return click_through


def line_chart(plt, x, y):
    # create a line chart, years on x-axis, gdp on y-axis
    plt.plot(x, y, color="#f39c12", marker="o", linestyle="solid")

    # add a title
    plt.title("Clicks movement")

    # add a label to the y-axis
    plt.ylabel("# clicks")
    plt.show()


def bar_chart(plt, x, y):
    # bars are by default width 0.8, so we'll add 0.1 to the left coordinates
    # so that each bar is centered
    xs = [i + 0.1 for i, _ in enumerate(x)]

    # plot bars with left x-coordinates [xs], heights [y]
    plt.bar(xs, y, color="#f39c12")
    plt.title("Clicks statistic")

    # label x-axis with color names at bar centers
    plt.xticks([i + 0.5 for i, _ in enumerate(x)], x)

    plt.show()


def cal_clicks(df):
    df["clicks"] = df["interactions"].map(lambda a: cal_click_through(a))
    return df


def cal_moving_average(df):
    df_clicks = df["clicks"]

    # calculating MA5: Moving Average over a 5-days, 12-days, 26-days
    ma5 = df_clicks.rolling(window=5).mean()
    ma12 = df_clicks.rolling(window=12).mean()
    ma26 = df_clicks.rolling(window=26).mean()

    # calculating MACD: Moving Average Convergence/ Divergence
    macd = ma12.subtract(ma26)

    # updating columns
    df["macd"] = macd
    df["ma5"] = ma5

    return df


def cal_rsi(df):
    # Get the difference in price from previous step
    delta = df["clicks"].diff()

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the SMA
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()

    # Calculate the RSI based on SMA
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    df["rsi"] = RSI

    return df


def return_decision(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return val


def to_epoch(time_param):
    date = time.strptime(time_param, "%Y%m%d%H")
    epoch = calendar.timegm(date)
    return epoch


if __name__ == "__main__":
    t_start = time.time()
    print "------------------------------------------------"
    print " %s - ETL clicks data" % datetime.datetime.now()
    print "------------------------------------------------"

    # parse terminal params
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_hour", required=True, help="Start hour YYYYMMDDHH (UTC).")
    ap.add_argument("-e", "--end_hour", required=True, help="End hour YYYYMMDDHH (UTC).")
    ap.add_argument("-w", "--widget_ids", required=True, help="Widget ID.")
    ap.add_argument("-a", "--adunit_ids", required=True, help="Adunit ID.")
    args = vars(ap.parse_args())

    # connect to databases
    conn = MongoClient("mongodb://localhost:27017/")
    db_brand_display_analytics = conn["brand_display_analytics"]

    # params
    widget_ids = args["widget_ids"].split(",")
    adunit_ids = args["adunit_ids"].split(",")
    start_hour = to_epoch(args["start_hour"])
    end_hour = to_epoch(args["end_hour"])
    coll_brand_display_analytics = db_brand_display_analytics["brand_display" + "_05_2017"]

    idx = 0
    for item in widget_ids:
        print "* Process widget:", item

        # query report
        num_items = 0
        cursor_report = coll_brand_display_analytics.find({
            "widgetId": item,
            "date": {
                "$gte": start_hour,
                "$lt": end_hour
            },
            "section": adunit_ids[idx]
        })
        # next item
        idx += 1

        # transforming data
        df_report_items = pd.DataFrame(list(cursor_report))
        df_report_items = df_report_items.sort_values(by=["date"], ascending=[1])

        # calculating clicks and moving average
        df_report_items = cal_clicks(df_report_items)
        df_report_items = cal_moving_average(df_report_items)

        # ROC: Rate of Change
        df_report_items["roc"] = df_report_items["clicks"].pct_change(periods=12)

        # RSI
        df_report_items = cal_rsi(df_report_items)

        # decision
        df_report_items["decision"] = df_report_items["clicks"].diff().map(lambda a: return_decision(a))

        df_out = df_report_items[["macd", "ma5", "roc", "rsi", "decision"]].loc[
            (pd.notnull(df_report_items["macd"])) &
            (df_report_items["roc"] != float("inf")) &
            (df_report_items["rsi"] != float("-inf"))
            ]

        # save to CSV file
        data_name = "data/" + item + ".csv"
        df_out["decision"] = df_out["decision"].astype(int)
        df_out.to_csv(path_or_buf=data_name, index=False)

    print " %s * DONE After * %s" % (datetime.datetime.now(), time_diff_str(t_start, time.time()))
