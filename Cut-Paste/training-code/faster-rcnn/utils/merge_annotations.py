""" Merge annotations from COCO and COCOstuff, i.e., stuff and things """

import json
from collections import OrderedDict

# set annotation directory
annotations_dir = "/media/data/philipp_data/COCOstuff/annotations"

# merge training annotations
with open(annotations_dir + "/instances_train2017.json") as f:
    train = json.load(f, object_pairs_hook=OrderedDict)

with open(annotations_dir + "/stuff_train2017.json") as f:
    stuff_train = json.load(f, object_pairs_hook=OrderedDict)

train["categories"].extend(stuff_train["categories"])
train["annotations"].extend(stuff_train["annotations"])
assert len(train["categories"]) == 172,  "should have 172 categories after merge"

with open(annotations_dir + "/train.json", "w") as f:
  json.dump(train, f)

# merge validation annotations
with open(annotations_dir + "/instances_val2017.json") as f:
    val = json.load(f, object_pairs_hook=OrderedDict)

with open(annotations_dir + "/stuff_val2017.json") as f:
    stuff_val = json.load(f, object_pairs_hook=OrderedDict)

val["categories"].extend(stuff_val["categories"])
val["annotations"].extend(stuff_val["annotations"])
assert len(val["categories"]) == 172,  "should have 172 categories after merge"

with open(annotations_dir + "/val.json", "w") as f:
  json.dump(val, f)