clear all; close all; clc;

load(['Mat/VHhumanStats_gravity.mat']);
Gravity = VHhumanStats;
load(['Mat/VHhumanStats.mat']);
Normal = VHhumanStats;
ImgRoot = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/';
selectedG = [];
selectedN = [];

Normal_classid = extractfield(Normal, 'classid');
Normal_whereapt = extractfield(Normal, 'whereapt');
Normal_roomid = extractfield(Normal, 'roomid');
Normal_surfaceid = extractfield(Normal, 'surfaceid');

for i = 1:length(Gravity)
    i
    classid = Gravity(i).classid;
    aptid = Gravity(i).whereapt;
    roomid = Gravity(i).roomid;
    surfaceid = Gravity(i).surfaceid;
    Gimg = imread([ImgRoot char(Gravity(i).imgname)]);
    
    ind = find(Normal_classid == classid & Normal_whereapt == aptid & Normal_roomid == roomid & Normal_surfaceid == surfaceid);
    
    for j = ind
        
        Nimg = imread([ImgRoot char(Normal(j).imgname)]);       
        %dist = sum(sum(sum(Nimg - Gimg)));
        %display(dist)      
        %if dist < 200000000
            selectedG = [selectedG i];
            selectedN = [selectedN j];
%             subplot(1,2,1);
%             imshow(Gimg);
%             subplot(1,2,2);
%             imshow(Nimg);
%             drawnow;
%             pause;
        %end
        
    end
    
    
    
end

selected = [selectedG; selectedN];
save(['Mat/OneToOne_gravity_normal.mat'], 'selected');
