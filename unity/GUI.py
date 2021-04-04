#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:08:29 2020

@author: mengmi
"""

# importing easygui module 
from easygui import *
import glob
import sys

rootdir = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/'

# What do you want to do?
msg ="What do you want to do?"
title = "Image Selector GUI"
choices = ["Classify Images", "See Good Images", "See Bad Images"]
choice = choicebox(msg, title, choices)

if choice == "See Good Images":
	# Choose an apartment
	msg ="Which apartment do you want to see?"
	title = "Choose an apartment"
	choices = ["0", "1", "2", "3", "4", "5", "6", "other"]
	choice = choicebox(msg, title, choices)
	apartment_name = "apartment_" + str(choice)

	# Load all good images from that apartment
	file_good = open(apartment_name + "_good.txt", "r") 
	good_images = file_good.readlines()


	if good_images == []:
		msg = msgbox("You need to have a file named " + apartment_name + "_good.txt in order to use this option.", "Warning")
		sys.exit()

	# Remove newlines
	good_images = map(lambda s: s.strip(), good_images)

	# Show all of the images
	for filename in good_images:
		output = buttonbox("good images", "Good Images", image = filename, choices = ["Next", "Cancel"])
		if output == "Cancel":
			break


elif choice == "See Bad Images":
	# Choose an apartment
	msg ="Which apartment do you want to see?"
	title = "Choose an apartment"
	choices = ["0", "1", "2", "3", "4", "5", "6", "other"]
	choice = choicebox(msg, title, choices)
	apartment_name = "apartment_" + str(choice)

	# Load all bad images from that apartment
	file_bad = open(apartment_name + "_bad.txt", "r") 
	bad_images = file_bad.readlines()


	if bad_images == []:
		msg = msgbox("You need to have a file named " + apartment_name + "_bad.txt in order to use this option.", "Warning")
		sys.exit()

	# Remove newlines
	bad_images = map(lambda s: s.strip(), bad_images)

	# Show all of the images
	for filename in bad_images:
		output = buttonbox("bad images", "Bad Images", image = filename, choices = ["Next", "Cancel"])
		if output == "Cancel":
			break




elif choice == "Classify Images":
	# Choose an apartment
	msg ="Which apartment do you want to review?"
	title = "Choose an apartment"
	choices = ["0", "1", "2", "3", "4", "5", "6", "other"]
	choice = choicebox(msg, title, choices)
	apartment_name = rootdir + "stimulus_size_ori_2/apartment_" + str(choice)


	# Load all of the images from that apartment
	all_images = glob.glob(apartment_name + '/*.png')

	if all_images == []:
		msg = msgbox("You need to have a folder named " + apartment_name + " in order to use this option.", "Warning")
		sys.exit()

	# Prepare the buttonbox
	text = "Should this image be included in the dataset"
	title = "Image Selector"
	button_list = ["Good", "Bad", "Cancel"] 
	total, good_cnt, bad_cnt = 0, 0, 0

	# Open files to write the results in
	file_good = open(apartment_name + "_good.txt", "w") 
	file_bad = open(apartment_name + "_bad.txt", "w") 


	# Go through all of the images in the chosen apartment
	for filename in all_images:

		button_not_clicked = True
		cancel = False

		while(button_not_clicked):

			output = buttonbox(text, title, image = filename, choices = button_list)
			button_not_clicked = False
			if output == "Good":
				good_cnt += 1
				file_good.write(filename + '\n') 

			elif output == "Bad":
				bad_cnt += 1
				file_bad.write(filename + '\n') 

			elif output == "Cancel":
				cancel = True

			else:
				button_not_clicked = True
				msg = msgbox("Please select one of the buttons!", "Try again") 


		if cancel:
			break
		else:
			total += 1

	# Show summary statistics after going through all of the images
	title = "Summary Statistics"
	message = "Total number of images processed: " + str(total) + "\n" + "Number of good images: " + str(good_cnt) + "\n" + "Number of bad images: " + str(bad_cnt) + "\n" + "Ratio accepted: " + str(good_cnt/total)  
	msg = msgbox(message, title) 

	# Close all files
	file_good.close()
	file_bad.close() 



