import os
from multiprocessing import Pool
from xml.dom import minidom

from bs4 import BeautifulSoup


def save_files(infos):
    (xml_path, saved_file) = infos

    if os.path.isfile(saved_file):
        os.unlink(saved_file)

    try:
        # parse xml file and get all articles
        doc = minidom.parse(xml_path)
        articles = doc.getElementsByTagName("article")

        # inspecting some stats
        total = len(articles)
        print "Processing", xml_path, total

        # get content from article and save to new file
        for art in articles:
            content = art.getElementsByTagName("content")[0]
            soup = BeautifulSoup(content.firstChild.data, "html5lib")
            text = soup.get_text().strip()

            # save to new file
            with open(saved_file, "a") as text_file:
                text_file.write(text.encode("utf8") + "\n")
    except Exception as e:
        print xml_path
        print "Error:", e


if __name__ == "__main__":
    total_articles = 0
    total_error = 0
    dirname = "/Users/hongong/Downloads/baomoi_articles"
    dir_sentences = "/Users/hongong/Downloads/sentences/"

    list_files = []
    for file_name in os.listdir(dirname):
        # get xml path, unlink before generating new content
        xml_path = os.path.join(dirname, file_name)
        saved_file = dir_sentences + file_name.split(".")[0] + ".txt"
        list_files.append((xml_path, saved_file))

    p = Pool(16)
    p.map(save_files, list_files)
