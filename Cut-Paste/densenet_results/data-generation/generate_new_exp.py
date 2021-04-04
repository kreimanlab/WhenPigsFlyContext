import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd





exp_files = ["test_expA_Color_oriimg.txt", "test_expH_Color_oriimg.txt"]
label_files = ["test_expA_Color_label.txt", "test_expH_Color_label.txt"]

all_images = {}

for file_name, label_file in zip(exp_files, label_files):
	f = open(file_name, "r")
	l = open(label_file, "r")
	for x, x_label in zip(f,l):
		all_images[x[:-16]] = x_label.strip('\n')
	f.close()

print(len(all_images))

# Experiment I is based on experiment A
f_ori = open("expIJ/test_expI_Color_oriimg.txt", "w")
f_bin = open("expIJ/test_expI_Color_binimg.txt", "w")
f_label = open("expIJ/test_expI_Color_label.txt", "w")
for file_name in all_images:

	f_ori.write(file_name + "2_imgtype_2.jpg\n")
	f_ori.write(file_name + "2_imgtype_8.jpg\n")
	f_bin.write(file_name + "1_binarybdbox.jpg\n")
	f_bin.write(file_name + "1_binarybdbox.jpg\n")
	f_label.write(all_images[file_name] + "\n")
	f_label.write(all_images[file_name] + "\n")

f_ori.close()
f_bin.close()
f_label.close()


# Experiment J is based on experiment H
f_ori = open("expIJ/test_expJ_Color_oriimg.txt", "w")
f_bin = open("expIJ/test_expJ_Color_binimg.txt", "w")
f_label = open("expIJ/test_expJ_Color_label.txt", "w")
for file_name in all_images:

	f_ori.write(file_name + "2_imgtype_1.jpg\n")
	f_ori.write(file_name + "2_imgtype_2.jpg\n")
	f_bin.write(file_name + "1_binarybdbox.jpg\n")
	f_bin.write(file_name + "1_binarybdbox.jpg\n")
	f_label.write(all_images[file_name] + "\n")
	f_label.write(all_images[file_name] + "\n")

f_ori.close()
f_bin.close()
f_label.close()