clear all; close all; clc;

load(['Mat/VHhumanStats_gravity.mat']);
classid = extractfield(VHhumanStats, 'classid');
for i = 1:36
    if length(find(classid == i)) < 50
        filtered_class{i}
    end
end