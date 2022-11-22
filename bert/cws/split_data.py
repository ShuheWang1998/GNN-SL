from lib2to3.pgen2 import token
import os

def split_ctb5(dir_, save_dir):
    train_file = open(os.path.join(save_dir, "train.txt"), "w", encoding='utf-8')
    dev_file = open(os.path.join(save_dir, "dev.txt"), "w", encoding='utf-8')
    test_file = open(os.path.join(save_dir, "test.txt"), "w", encoding='utf-8')

    for root, dirs, files in os.walk(dir_):
        for name in files:
            #print(name)
            name_index = int(name[5:-4])
            file = open(os.path.join(dir_, name), "r", encoding='iso-8859-1')
            flag = False
            for line in file:
                if flag:
                    flag = False
                    try:
                        line = line.encode('iso-8859-1').decode("gbk")
                    except Exception:
                        pass
                    else:
                        pass
                    if 301 <= name_index <= 325:
                        dev_file.write(line.strip()+'\n')
                    elif 271 <= name_index <= 300:
                        test_file.write(line.strip()+'\n')
                    else:
                        train_file.write(line.strip()+'\n')
                if len(line) > 5 and line[:5] == "<S ID":
                    flag = True

def clean_ctb5(dir_):
    prefixs = ["test", "dev", "train"]
    for prefix in prefixs:
        file = open(os.path.join(dir_, f"{prefix}.txt"), "r", encoding='utf-8')
        out_file = open(os.path.join(dir_, f"{prefix}.clean.txt"), "w", encoding='utf-8')
        for line in file:
            line = line.strip().split(" ")
            if len(line) == 0:
                continue
            string_ = ""
            for token in line:
                if token == "_-NONE-":
                    continue
                string_ += token + " "
            out_file.write(string_.strip()+'\n')
        file.close()
        out_file.close()

def split_ctb6(dir_, save_dir):
    train_file = open(os.path.join(save_dir, "train.txt"), "w", encoding='utf-8')
    dev_file = open(os.path.join(save_dir, "dev.txt"), "w", encoding='utf-8')
    test_file = open(os.path.join(save_dir, "test.txt"), "w", encoding='utf-8')
    
    def get_type(name_index):
        if 41 <= name_index <= 80 or 1120 <= name_index <= 1129 or 2140 <= name_index <= 2159 or 2280 <= name_index <= 2294:
            return "dev"
        if 2550 <= name_index <= 2569 or 2775 <= name_index <= 2799 or 3080 <= name_index <= 3109:
            return "dev"
        if 1 <= name_index <= 40 or 901 <= name_index <= 931 or name_index == 1018 or name_index == 1020 or name_index == 1036:
            return "test"
        if name_index == 1044 or 1060 <= name_index <= 1061 or name_index == 1072 or 1118 <= name_index <= 1119 or name_index == 1132:
            return "test"
        if 1141 <= name_index <= 1142 or name_index == 1148 or 2165 <= name_index <= 2180 or 2295 <= name_index <= 2310:
            return "test"
        if 2570 <= name_index <= 2602 or 2800 <= name_index <= 2819 or 3110 <= name_index <= 3145:
            return "test"
        return "train"

    for root, dirs, files in os.walk(dir_):
        for name in files:
            #print(name)
            name_index = int(name[5:-4])
            file = open(os.path.join(dir_, name), "r", encoding='utf-8')
            for line in file:
                line = line.strip()
                if len(line) == 0 or (len(line) >= 2 and line[0] == '<' and line[-1] == '>'):
                    continue
                type_ = get_type(name_index)
                if type_ == "train":
                    train_file.write(line+'\n')
                elif type_ == "dev":
                    dev_file.write(line+'\n')
                else:
                    test_file.write(line+'\n')

def clean_ctb6(dir_):
    prefixs = ["test", "dev", "train"]
    for prefix in prefixs:
        file = open(os.path.join(dir_, f"{prefix}.txt"), "r", encoding='utf-8')
        out_file = open(os.path.join(dir_, f"{prefix}.clean.txt"), "w", encoding='utf-8')
        for line in file:
            line = line.strip().split(" ")
            if len(line) == 0:
                continue
            string_ = ""
            for token in line:
                if token == "_-NONE-" or len(token.strip().split("_")) != 2:
                    continue
                string_ += token + " "
            out_file.write(string_.strip()+'\n')
        file.close()
        out_file.close()

def clean_ud1_4(dir_, save_dir):
    prefixs = ["dev", "test", "train"]
    for prefix in prefixs:
        path = os.path.join(dir_, f"zh-ud-{prefix}.conllu")
        file = open(path, "r", encoding='utf-8')
        out_file = open(os.path.join(save_dir, f"{prefix}.txt"), "w", encoding='utf-8')
        string_ = ""
        for line in file:
            line = line.strip()
            if line == "":
                string_ = string_.strip()
                if len(string_) != 0:
                    out_file.write(string_+'\n')
                string_ = ""
                continue
            line = line.split("\t")
            token_ = line[1] + "_"
            token_ += line[4]
            string_ += token_ + " "
        file.close()
        out_file.close()

def clean_wsj(dir_, save_dir):
    bra_ = {}
    bra_["-LCB-"] = "["
    bra_["-LRB-"] = "("
    bra_["-RCB-"] = "]"
    bra_["-RRB-"] = ")"
    prefixs = ["dev", "test", "train"]
    for prefix in prefixs:
        path = os.path.join(dir_, f"{prefix}.tsv")
        file = open(path, "r", encoding='utf-8')
        out_file = open(os.path.join(save_dir, f"{prefix}.clean.txt"), "w", encoding='utf-8')
        string_ = ""
        for line in file:
            line = line.strip()
            if line == "":
                string_ = string_.strip()
                if len(string_) != 0:
                    out_file.write(string_+'\n')
                string_ = ""
                continue
            line = line.split("\t")
            token_ = ""
            if line[0] in bra_:
                token_ += bra_[line[0]]
            else:
                token_ += line[0]
            token_ += "_"
            if line[1] in bra_:
                token_ += bra_[line[1]]
            else:
                token_ += line[1]
            string_ += token_ + " "
        file.close()
        out_file.close()
    
if __name__ == '__main__':
    #split_ctb5("/data2/wangshuhe/gnn_ner/pos_data/LDC2005T01/data/postagged",
    #           "/data2/wangshuhe/gnn_ner/pos_data/ctb5")
    #clean_ctb5("/data2/wangshuhe/gnn_ner/pos_data/ctb5")
    #split_ctb6("/data2/wangshuhe/gnn_ner/pos_data/LDC2007T36/data/utf8/postagged",
    #           "/data2/wangshuhe/gnn_ner/pos_data/ctb6")
    #clean_ctb6("/data2/wangshuhe/gnn_ner/pos_data/ctb6")
    #clean_ud1_4("/data2/wangshuhe/gnn_ner/pos_data/ud1.4/ud-treebanks-v1.4/UD_Chinese",
    #           "/data2/wangshuhe/gnn_ner/pos_data/ud1.4/data")
    clean_wsj("/data2/wangshuhe/gnn_ner/pos_data/wsj",
               "/data2/wangshuhe/gnn_ner/pos_data/wsj")