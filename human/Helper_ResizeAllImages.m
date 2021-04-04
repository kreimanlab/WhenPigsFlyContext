clear all; close all; clc;

NumApt = 7;
ImgRootDir = '/home/mengmi/Projects/Proj_context3/VirtualHome/gitsource/demo/stimulus/';

for aptid = [1:2] %NumApt]
    aptid
    imgdc = dir([ImgRootDir 'apartment_' num2str(aptid-1) '/*.png']);
    
    for j = 1:length(imgdc)
        img = imread([ImgRootDir 'apartment_' num2str(aptid-1) '/' imgdc(j).name]);
        img = imresize(img, [1024 2048]);
        imwrite(img,[ImgRootDir 'apartment_' num2str(aptid-1) '/' imgdc(j).name]);
    end
end