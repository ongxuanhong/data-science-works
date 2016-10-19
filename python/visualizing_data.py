import matplotlib.pyplot as plt
import numpy as np


def my_line_chart(plt):
    years = ["1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997",
             "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010",
             "2011", "2012", "2013", "2014", "2015"]
    gdp = [14094688429, 26336617862, 36658108169, 25423812494, 6293304847, 6471740486,
           9613369553, 9866990096, 13180954014, 16286434094, 20736163915, 24657470331,
           26843701136, 27209601995, 28683658004, 33640085727, 35291349277, 37947904054,
           42717072777, 49424107709, 57633255739, 66371664817, 77414425532, 99130304099,
           106014600963, 115931749904, 135539487317, 155820001920, 171222025117, 186204652922,
           193599379094]

    # create a line chart, years on x-axis, gdp on y-axis
    plt.plot(years, gdp, color='#f39c12', marker='o', linestyle='solid')

    # add a title
    plt.title("Vietnam GDP")

    # add a label to the y-axis
    plt.ylabel("Billions of $")
    plt.show()


def my_bar_chart(plt):
    color_names = ["Emerald", "Green Sea", "Midnight Blue", "Carrot", "Peter River"]
    colors = ["#2ecc71", "#16a085", "#2c3e50", "#e67e22", "#3498db"]
    num_favorite = [5, 11, 3, 8, 10]

    # bars are by default width 0.8, so we'll add 0.1 to the left coordinates
    # so that each bar is centered
    xs = [i + 0.1 for i, _ in enumerate(color_names)]

    # plot bars with left x-coordinates [xs], heights [num_favorite]
    plt.bar(xs, num_favorite, color=colors)
    plt.title("My Favorite Colors")

    # label x-axis with color names at bar centers
    plt.xticks([i + 0.5 for i, _ in enumerate(color_names)], color_names)

    plt.show()


def my_histogram(plt):
    data = []
    for i in range(100):
        data.append(np.random.randint(1, 11))

    plt.hist(data, bins=10, facecolor='#bdc3c7')

    plt.xlabel("Points")
    plt.ylabel("# of Students")
    plt.title("Results of the exam")
    plt.show()


def my_multi_line_charts(plt):
    bears = [10, 58, 85, 115, 139, 182]
    dolphins = [150, 75, 32, 14, 8, 5]
    whales = [80, 50, 100, 75, 90, 70]
    x = [0, 1, 2, 3, 4, 5]
    years = ["2009", "2010", "2011", "2012", "2013", "2014"]

    # we can make multiple calls to plt.plot 
    # to show multiple series on the same chart
    plt.plot(x, bears, '#16a085', marker='o', linewidth=3.0, label='Bears')
    plt.plot(x, dolphins, '#c0392b', marker='s', linewidth=3.0, label='Dolphins')
    plt.plot(x, whales, '#3498db', marker='^', linewidth=3.0, label='Whales')

    # because we've assigned labels to each series
    # we can get a legend for free
    # loc=9 means "top center"
    plt.legend(loc=9)
    plt.title("Number of animals each year")
    plt.xlabel("Years")
    plt.xticks(x, years)
    plt.show()


def my_scatter_plot(plt):
    sizes = [700, 650, 720, 630, 710, 640, 600, 640, 670]
    prices = [175, 170, 205, 120, 220, 130, 105, 145, 190]
    labels = ["$175", "$170", "$205", "$120", "$220", "$130", "$105", "$145", "$190"]

    plt.scatter(sizes, prices, marker='s', s=40, color='#2ecc71')

    # label each point
    for label, friend_count, minute_count in zip(labels, sizes, prices):
        plt.annotate(label,
                     xy=(friend_count, minute_count),  # put the label with its point
                     xytext=(5, -5),  # but slightly offset
                     textcoords='offset points')

    plt.title("House prices")
    plt.xlabel("Size in m2")
    plt.ylabel("Thousand $")
    plt.show()


def my_pie_chart(plt):
    data = [0.5, 0.26, 0.11, 0.04, 0.02, 0.02, 0.01, 0.04]
    smart_phone = ["Apple", "Samsung", "LG", "Motorola", "HTC", "Nokia", "Amazon", "Other"]
    colors = ["#ecf0f1", "#3498db", "#e67e22", "#1abc9c", "#bdc3c7", "#8e44ad", "#f39c12", "#2c3e50"]

    plt.pie(data, labels=smart_phone, colors=colors, autopct='%1.1f%%',
            startangle=-90, pctdistance=0.9, labeldistance=1.2)

    # make sure pie is a circle and not an oval
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    my_line_chart(plt)

    my_bar_chart(plt)

    my_histogram(plt)

    my_multi_line_charts(plt)

    my_scatter_plot(plt)

    my_pie_chart(plt)
