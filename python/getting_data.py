import csv
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
from twython import Twython


def print_data(ma_ck, kl, gia, delta):
    print ma_ck, "#", kl, "#", gia, "#", delta


####
#
# Oreilly
#
####

def is_video(td):
    """it's a video if it has exactly one pricelabel, and if
    the stripped text inside that pricelabel starts with 'Video'"""
    price_labels = td('span', 'pricelabel')
    return (len(price_labels) == 1 and
            price_labels[0].text.strip().startswith("Video"))


def book_info(td):
    """given a BeautifulSoup <td> Tag representing a book,
    extract the book's details and return a dict"""

    title = td.find("div", "thumbheader").a.text
    by_author = td.find('div', 'AuthorName').text
    authors = [x.strip() for x in re.sub("^By ", "", by_author).split(",")]
    isbn_link = td.find("div", "thumbheader").a.get("href")
    isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
    date = td.find("span", "directorydate").text.strip()

    return {
        "title": title,
        "authors": authors,
        "isbn": isbn,
        "date": date
    }


def scrape(num_pages=10):
    base_url = "http://shop.oreilly.com/category/browse-subjects/data.do?sortby=publicationDate&page="

    books = []

    for page_num in range(1, num_pages + 1):
        print "souping page", page_num
        url = base_url + str(page_num)
        soup = BeautifulSoup(requests.get(url).text, 'lxml')

        for td in soup('td', 'thumbtext'):
            if not is_video(td):
                books.append(book_info(td))

    return books


def get_year(book):
    """book["date"] looks like 'November 2014' so we need to
    split on the space and then take the second piece"""
    return int(book["date"].split()[1])


def plot_years(plt, books):
    # 2014 is the last complete year of data (when I ran this)
    year_counts = Counter(get_year(book) for book in books
                          if get_year(book) <= 2016)

    years = sorted(year_counts)
    book_counts = [year_counts[year] for year in years]
    plt.bar([x - 0.5 for x in years], book_counts)
    plt.xlabel("year")
    plt.ylabel("# of data books")
    plt.title("Data is Big!")
    plt.show()


####
#
# Twitter
#
####

# fill these in if you want to use the code
CONSUMER_KEY = "JeuEwD5RJiBbxiw9jTMBYBEmU"
CONSUMER_SECRET = "xRcmv8AMnSSMwq875HiP1SKFfGw51M97BvVH341yckPY3iilCu"
ACCESS_TOKEN = "47319754-NL1AIh9PBomIVsJe5HXB9vjE5y1rjwZFYUQx0odzo"
ACCESS_TOKEN_SECRET = "kcq7ER8UZSykDomPn9lYdh5DAafndvp73PzSfykTq0Kp7"


def call_twitter_search_api():
    twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)

    # search for tweets containing the phrase "data science"
    for status in twitter.search(q='"data science"')["statuses"]:
        user = status["user"]["screen_name"].encode('utf-8')
        text = status["text"].encode('utf-8')
        print user, ":", text
        print


if __name__ == "__main__":
    print "# Data from: http://s.cafef.vn/du-lieu.chn"
    print "## TAB delimited stock prices"

    with open('data/tab_delimited_stock_prices.tsv', 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            ma_ck = row[0]
            kl = row[1]
            gia = float(row[2])
            delta = row[3]
            print_data(ma_ck, kl, gia, delta)

    print

    print "## COLON delimited stock prices"
    with open('data/colon_delimited_stock_prices.csv', 'rb') as f:
        reader = csv.DictReader(f, delimiter=':')
        for row in reader:
            ma_ck = row["MA_CK"]
            kl = row["KL"]
            gia = float(row["GIA"])
            delta = row["DELTA"]
            print_data(ma_ck, kl, gia, delta)

    print

    print "## WRITING out comma_delimited_stock_prices.csv"
    today_prices = {'VCF': 152.4, 'VAF': 13.3, 'ATA': 0.8}
    with open('data/comma_delimited_stock_prices.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        for stock, price in today_prices.items():
            writer.writerow([stock, price])

    print "## BeautifulSoup"
    html = requests.get("https://www.google.com").text
    soup = BeautifulSoup(html, "lxml")
    print soup
    print

    print "## PARSING json"
    # parse the JSON to create a Python object
    with open("data/colors.json") as json_data:
        document = json.load(json_data)
        print "Getting blue value:", document["blue"]

    print

    print "## GitHub API"
    endpoint = "https://api.github.com/users/ongxuanhong/repos"
    repos = json.loads(requests.get(endpoint).text)

    dates = [parse(repo["created_at"]) for repo in repos]
    month_counts = Counter(date.month for date in dates)
    weekday_counts = Counter(date.weekday() for date in dates)

    print "dates", [d.strftime("%d/%m/%y") for d in dates]
    print "month_counts", month_counts
    print "weekday_count", weekday_counts

    last_5_repositories = sorted(repos,
                                 key=lambda r: r["created_at"],
                                 reverse=True)[:5]

    print "last five repos", [repo["name"]
                              for repo in last_5_repositories]
    print

    print "## Oreilly books"
    books = scrape()
    plot_years(plt, books)
    print

    print "## Twitter search"
    call_twitter_search_api()
