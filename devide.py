import os


if __name__ == "__main__":
    file_name = "all_sentences.txt"
    dir_name = "separated" 
    for i, line in enumerate(open(file_name, "r", encoding="utf-8")):
        with open(os.path.join(dir_name, "{}.txt".format(i + 1)), "w", encoding="utf-8") as f:
            f.writelines(line.strip()) 
            f.close()