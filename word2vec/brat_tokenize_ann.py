if __name__ == "__main__":
    file_wseg = "data/sample_tokenize.txt.sent.tkn.wseg"
    file_ann = "data/sample_tokenize.ann"
    out_str = ""
    with open(file_wseg, "r") as text_file:
        curr_pos = 0
        curr_tag_id = 1
        lines = text_file.readlines()
        for line in lines:
            words = line.split(" ")
            for word in words:
                sub_words = word.split("_")
                for idx, sub in enumerate(sub_words):
                    begin_span = curr_pos
                    end_span = curr_pos + len(sub.decode("utf-8"))
                    if idx == 0:
                        tag_name = "B_W"
                    else:
                        tag_name = "I_W"

                    out_str += "T" + str(curr_tag_id) + "\t" + tag_name + " " + str(begin_span) + " " + str(
                        end_span) + "\t" + sub + "\n"
                    curr_pos = end_span + 1
                    curr_tag_id += 1

    with open(file_ann, "w") as out_ann:
        out_ann.writelines(out_str)
