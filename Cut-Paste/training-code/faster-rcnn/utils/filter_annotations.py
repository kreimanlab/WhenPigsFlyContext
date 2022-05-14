""" Filter annotations from COCOstuff to select only classes that overlap with UnRel"""

import os.path
import pathlib
import json
from collections import OrderedDict

# set annotation directory
annotations_dir = "/media/data/philipp_data/COCOstuff/annotations"
annotations_outdir = "/media/data/philipp_data/COCOstuff/annotations_UnRel_compatible"
filter_classes_file = "/media/data/philipp_data/UnRel_test/classes_COCOstuff_format.txt"

pathlib.Path(annotations_outdir).mkdir() 

# load labels to be filtered
labels = set()
assert os.path.isfile(filter_classes_file), "Specify the path to a file with the classes to be filtered."
with open(filter_classes_file) as f:
    for line in f:
        labels.add(line.strip("\n"))
        print(line.strip("\n") + "\t")

## filter training labels
with open(annotations_dir + "/train.json") as f:
    train = json.load(f, object_pairs_hook=OrderedDict)

# filter categories
new_categories = []
keeper_indices = set()
for c in train["categories"]:
    if c["name"] in labels:
        new_categories.append(c)
        keeper_indices.add(c["id"])

train["categories"] = new_categories

# filter annotations
new_annotations = []
for a in train["annotations"]:
    if a["category_id"] in keeper_indices:
        new_annotations.append(a)

train["annotations"] = new_annotations

with open(annotations_outdir + "/train.json", "w") as f:
  json.dump(train, f)


## Filter validation labels

with open(annotations_dir + "/val.json") as f:
    val = json.load(f, object_pairs_hook=OrderedDict)

# filter categories
new_categories = []
keeper_indices = set()
for c in val["categories"]:
    if c["name"] in labels:
        new_categories.append(c)
        keeper_indices.add(c["id"])

val["categories"] = new_categories

# filter annotations
new_annotations = []
for a in val["annotations"]:
    if a["category_id"] in keeper_indices:
        new_annotations.append(a)

val["annotations"] = new_annotations

with open(annotations_outdir + "/val.json", "w") as f:
  json.dump(val, f)
