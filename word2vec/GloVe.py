import argparse
import subprocess
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--vocab-min", type=int, default=10)
    parser.add_argument("--size", type=int)
    parser.add_argument("--iter", type=int, default=30)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--out", type=str)
    opt = parser.parse_args()
    out_dir, out_file = os.path.split(opt.out)
    GLOVE_PATH = "/home/sekine/glove/build/"
    GLOVE_PATH = "/Users/hiroto/glove/build/"

    shellscripts = [
        GLOVE_PATH + "vocab_count -min-count {} < {} > ./vocab.txt".format(opt.vocab_min, opt.input),
        GLOVE_PATH + "cooccur -memory 4 -vocab-file vocab.txt -window-size {} < {} > ./cooccur.bin".format(opt.window, opt.input),
        GLOVE_PATH + "shuffle -memory 4 < ./cooccur.bin > ./cooccur.shuff.bin",
        GLOVE_PATH + "glove -save-file vectors -threads 8 -input-file ./cooccur.shuff.bin -x-max 100 -iter {} -vector-size {} -binary 2 -vocab-file ./vocab.txt".format(opt.iter, opt.size),
        "mv ./vectors.txt {}".format(opt.out), 
        "mv ./vocab.txt {}".format(opt.out + ".vocab"), 
        "rm cooccur.bin cooccur.shuff.bin vectors.bin"
    ]

    for script in shellscripts:
        print("=========================================")
        print(script)
        subprocess.call(script, shell=True)
    print("Glove Finished!!!")