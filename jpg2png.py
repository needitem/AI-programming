import os
import sys
from PIL import Image


def jpg2png(src_dir):
    for file in os.listdir(src_dir):
        if file.endswith(".jpg"):
            img = Image.open(os.path.join(src_dir, file))
            img.save(os.path.join(src_dir, file.replace(".jpg", ".png")))


if __name__ == "__main__":
    jpg2png("Data/Apple_Good")
    jpg2png("Data/Apple_Bad")
    print("Done")
