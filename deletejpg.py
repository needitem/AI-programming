import os, sys

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def deletejpg(src_dir):
    count = 0
    for file in os.listdir(src_dir):
        if file.endswith(".jpg"):
            os.remove(os.path.join(src_dir, file))
            count += 1
    print(f"Deleted {count} files")


if __name__ == "__main__":
    deletejpg("Data/train/good")
    deletejpg("Data/train/bad")
    print("Done")
