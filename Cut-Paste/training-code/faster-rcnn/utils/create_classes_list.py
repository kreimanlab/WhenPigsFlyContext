import os.path
import json

from collections import OrderedDict

with open("/media/data/philipp_data/COCOstuff/annotations/instances_val2017.json") as f:
    val = json.load(f, object_pairs_hook=OrderedDict)

with open("/media/data/philipp_data/COCOstuff/annotations/stuff_val2017.json") as f:
    stuff_val = json.load(f, object_pairs_hook=OrderedDict)

val["categories"].extend(stuff_val["categories"])
val["annotations"].extend(stuff_val["annotations"])
assert len(val["categories"]) == 172,  "should have 172 categories after merge"

with open('/media/data/philipp_data/COCOstuff/COCOstuff_172classes.txt', 'w') as f:
    for item in val["categories"]:
        f.write("%s\n" % item["name"])
