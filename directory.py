import os, sys
import random

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


good_apple = next(os.walk("Data/Apple_Good"))[2]
bad_apple = next(os.walk("Data/Apple_Bad"))[2]


def split_files(src_dir, ratio, test_dir, train_dir):
    # Get a list of all files in the source directory
    files = os.listdir(src_dir)
    count = int(len(files) * ratio)
    # Shuffle the list of files
    random.shuffle(files)

    for i in range(count):
        # Move the first `count` files to the test directory
        os.rename(os.path.join(src_dir, files[i]), os.path.join(test_dir, files[i]))

    for i in files[count:]:
        # Move the rest of the files to the train directory
        os.rename(os.path.join(src_dir, i), os.path.join(train_dir, i))


test_dir_good = "Data/test/good"
train_dir_good = "Data/train/good"
test_dir_bad = "Data/test/bad"
train_dir_bad = "Data/train/bad"

if not os.path.exists(test_dir_good):
    os.makedirs(test_dir_good)
if not os.path.exists(train_dir_good):
    os.makedirs(train_dir_good)
if not os.path.exists(test_dir_bad):
    os.makedirs(test_dir_bad)
if not os.path.exists(train_dir_bad):
    os.makedirs(train_dir_bad)


def shuffle_files():
    for i in os.listdir(test_dir_good):
        os.rename(os.path.join(test_dir_good, i), os.path.join("Data/Apple_Good", i))
    for i in os.listdir(train_dir_good):
        os.rename(os.path.join(train_dir_good, i), os.path.join("Data/Apple_Good", i))
    for i in os.listdir(test_dir_bad):
        os.rename(os.path.join(test_dir_bad, i), os.path.join("Data/Apple_Bad", i))
    for i in os.listdir(train_dir_bad):
        os.rename(os.path.join(train_dir_bad, i), os.path.join("Data/Apple_Bad", i))
    split_files("Data/Apple_Good", 0.2, test_dir_good, train_dir_good)
    split_files("Data/Apple_Bad", 0.2, test_dir_bad, train_dir_bad)


shuffle_files()
