clear all; close all; clc;

dataset = 'expAnomaly';
% load(['Mat/mturk_GTlabel_compiled.mat']);
% header = 'http://kreimanlab.com/mengmiMturkHost/VirtualHome/GT2/';
% imgfolder = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/';

load(['Mat/mturk_expAnomaly_compiled.mat']);
%header = 'http://kreimanlab.com/mengmiMturkHost/Unrel/';
header = 'http://kreimanlab.com/mengmiMturkHost/VirtualHome/keyframe_VH_anomaly_gif/';
%imgfolder = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/matlab/Stimulus_unrel/';
imgfolder = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/matlab/Stimulus/keyframe_VH_anomaly/';

trialnum = 100;

for s = [1]

    SetNo = s;    
    mkdir(['/home/mengmi/Desktop/send/' dataset '/Set_' num2str(SetNo) ]);    

    counter = 1;
    for i = randperm(length(mturkData), length(mturkData))
        ans = mturkData(i).answer; 
%         if mturkData(i).subjvalid == 0
%             continue;
%         end

        responseList = extractfield(ans,'response');
        imgpresentid = extractfield(ans,'imageID');

        ind = randperm(length(responseList),1);
        res = responseList{ind};
        imgname = imgpresentid{ind};
        imgname = imgname(length(header)+5:end);
        
        imgname = ['trial_oriimg_' imgname(1:end-4) '.jpg'];

        img = imread([imgfolder imgname ]);
        RGB = insertText(img,[1 1],res,'FontSize',38,'BoxColor',...
        'r','BoxOpacity',0.4,'TextColor','white');
        RGB = imresize(RGB, [480 640]);
        imwrite(RGB, ['/home/mengmi/Desktop/send/' dataset '/Set_' num2str(SetNo) '/imgeg_' num2str(counter) '.jpg']);
        counter = counter+1;
        
        if counter > trialnum
            break;
        end
        
    end

end
