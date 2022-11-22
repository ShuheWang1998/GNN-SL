import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str)
parser.add_argument('--file-name', type=str)
parser.add_argument('--type', type=str, default="ner")
args = parser.parse_args()

if args.type == "ner":
    input_file = open(os.path.join(args.data_dir, "train."+args.file_name), "r")
    vis = {}
    for line in input_file:
        line = line.strip().split()
        if (len(line) == 0):
            continue
        vis[line[1]] = True
elif args.type == "pos":
    input_file = open(os.path.join(args.data_dir, "train."+args.file_name), "r")
    vis = {}
    vis["O"] = True
    idx_ = 0
    for line in input_file:
        line = line.strip().split("\t")
        for word in line[1].strip().split(" "):
            vis[word] = True
else:
    input_file = open(os.path.join(args.data_dir, "train."+args.file_name), "r")
    vis = {}
    idx_ = 0
    for line in input_file:
        line = line.strip().split(" ")
        print(idx_)
        idx_ += 1
        for token in line:
            vis[token.strip().split('_')[1]] = True

out_file = open(os.path.join(args.data_dir, "ner_labels.txt"), "w")
for key, _ in vis.items():
    out_file.write(key+'\n')