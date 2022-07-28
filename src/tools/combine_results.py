import json
import os
import sys
from pathlib import Path


def combine_results(folder, output_file="comb.json", has_lists=False):
    for root, subdirs, files in os.walk(folder):
        r = {}
        n_files = 0
        for file in files:
            if file.endswith(".json") and not file == output_file and not file.endswith("conf.json"):
                n_files += 1
                with open(os.path.join(root, file)) as ff:
                    fff = json.load(ff)
                    for key in fff:
                        if key != 'x':
                            if has_lists:
                                if key in r:
                                    r[key] += fff[key]
                                else:
                                    r[key] = fff[key]
                            else:
                                if key in r:
                                    r[key].append(fff[key])
                                else:
                                    r[key] = [fff[key]]
        print("{n} files combined".format(n=n_files))
        if len(r) > 0:
            Path(os.path.join(root)).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(root, output_file), 'w') as file:
                json.dump(r, file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        resultDir = sys.argv[1]
    else:
        resultDir = "results/TVDIS_JP/sbm/"
    combine_results(resultDir,has_lists=False)
