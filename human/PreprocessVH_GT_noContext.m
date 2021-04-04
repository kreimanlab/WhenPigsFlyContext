clear all; close all; clc;

%load('Mat/VHhumanStats.mat');
load('Mat/VHhumanStats_noContext.mat');

ImgDir = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/';
WriteDir = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/matlab/';

nameheader1 = 'stimulus_2';
nameheader2 = 'stimulus_2_ori';

%filter out missing segmentation mask and save to VHhumanStats_noContext.mat
% newstruct = [];
% for t = 1:length(VHhumanStats)
%     imgname = convertStringsToChars(VHhumanStats(t).imgname);
%     if strcmp(imgname(1:length(nameheader2)), nameheader2)
%         binname = strcat(ImgDir, nameheader2, '_seg', imgname(length(nameheader2)+1:end));
%     else
%         binname = strcat(ImgDir, nameheader1, '_seg', imgname(length(nameheader1)+1:end));
%     end
%     if exist(binname, 'file') == 2 && exist(strcat(ImgDir, imgname), 'file') == 2
%         VHhumanStats(t).segname = binname(length(ImgDir)+1:end);
%         newstruct = [newstruct VHhumanStats(t)];
%     end
% end
% VHhumanStats = newstruct;
% save(['Mat/VHhumanStats_noContext.mat'],'VHhumanStats');
% warning(['now run ComputeTargetObjSizeDVA.m']);

NumImg = length(VHhumanStats);
ScreenWidth = 1024;
ScreenHeight = 1280;
exptime = [500 1000 200 100]; %in millisecs
cf = 100; %this is the greatest common factor of experiment time
boundingboxWidth = 3;
keyframetype = 'Stimulus/keyframe_VH_NoContext'; %change this to keyframe_expB, etc

%clean and create two folders for the type of expeirment
%rmdir([keyframetype], 's');
%rmdir([keyframetype '_gif'], 's');
mkdir([WriteDir keyframetype]);
mkdir([WriteDir keyframetype '_gif']);

for t = 1:NumImg
    imgname = convertStringsToChars(VHhumanStats(t).imgname);
    img = imread(strcat(ImgDir, imgname));
    img = imresize(img, [ScreenWidth, ScreenHeight]);
    
    if length(size(img))~=3
        img = cat(3, img, img, img);
    end
    
    % convert to binary image
    if strcmp(imgname(1:length(nameheader2)), nameheader2)
        bin = imread(strcat(ImgDir, nameheader2, '_seg', imgname(length(nameheader2)+1:end)));
    else
        bin = imread(strcat(ImgDir, nameheader1, '_seg', imgname(length(nameheader1)+1:end)));
    end
    
    bin = imresize(bin,[1024 1280]);
    bin = im2bw(bin,0.5);
    
    
    %bboxp%os = [VHhumanStats(t).left+1 VHhumanStats(t).top+1 VHhumanStats(t).right-VHhumanStats(t).left VHhumanStats(t).bottom-VHhumanStats(t).top];
    %imgbbox = insertShape(img,'Rectangle',bboxpos,'LineWidth',boundingboxWidth,'Color','yellow');
    %bbox already on the image
    imgbbox = img;
    %bin = zeros([size(img,1) size(img,2)]);
    %bin(bboxpos(2):bboxpos(2)+bboxpos(4), bboxpos(1):bboxpos(1)+bboxpos(3)) = 255;
    
    %imshow(uint8(imgbin));    
    %imshow(imgbbox);
    % rescale to fit screen size; maintain aspect ratio
    RGB = imresize(imgbbox, [ScreenWidth NaN]);   
    RGBbin = imresize(bin, [ScreenWidth NaN]);
    if size(RGB,2) > ScreenHeight
        RGB = imresize(imgbbox, [NaN ScreenHeight]); 
        RGBbin = imresize(bin, [NaN ScreenHeight]);
    end
    img = RGB;
    bin = im2bw(RGBbin);
    [iw ih] = size(RGBbin);
    
    % paste them in center of screen
    screen = ones(ScreenWidth, ScreenHeight,3)*128;
    binscreen = ones(ScreenWidth, ScreenHeight,3)*0;
    if iw == ScreenWidth
        ileftx = floor(ScreenHeight/2) - floor(ih/2)+1; irightx = floor(ScreenHeight/2) - floor(ih/2)+size(img,2); ilefty = 1; irighty = ScreenWidth;
        screen(:,ileftx: irightx,:) = img;
        binscreen(:,ileftx: irightx,1) = bin;
        binscreen(:,ileftx: irightx,2) = bin;
        binscreen(:,ileftx: irightx,3) = bin;
        
    else
        ileftx = 1; irightx = ScreenHeight; ilefty = floor(ScreenWidth/2) - floor(iw/2)+1; irighty = floor(ScreenWidth/2) - floor(iw/2) + size(img,1);
        screen(ilefty:irighty,:,:) = img;
        binscreen(ilefty:irighty,:,1) = bin;
        binscreen(ilefty:irighty,:,2) = bin;
        binscreen(ilefty:irighty,:,3) = bin;
    end
    
    imwrite(binscreen,[WriteDir keyframetype '/trial_binimg_' num2str(t) '_' num2str(VHhumanStats(t).classid) ...
        '_' num2str(VHhumanStats(t).whereapt) '_' num2str(VHhumanStats(t).roomid) '.jpg']);
    
    % extract bounding box; x is horinzontal axis; y is vertical axis; (0,0) is
    % at top left corner of the image
    [row, col] = find(binscreen(:,:,1)==1);
    leftx = min(col);lefty = min(row);rightx = max(col);righty = max(row);
    ctrx = floor((leftx + rightx)/2);ctry = floor((lefty + righty)/2);
    oh = rightx - leftx;
    ow = righty - lefty; 
    
    img_silhouetteSP = randi([0 1], ScreenWidth, ScreenHeight)*128;
    img_silhouetteSP = cat(3, img_silhouetteSP, img_silhouetteSP, img_silhouetteSP);
    img_silhouette = ones(ScreenWidth, ScreenHeight,3)*128;
    img_silhouette(find(binscreen == 1)) = screen(find(binscreen == 1));
    img_silhouetteSP(find(binscreen == 1)) = screen(find(binscreen == 1));
    
    img_silhouette = uint8(img_silhouette);
    img_silhouette = insertShape(img_silhouette,'Rectangle',[leftx lefty rightx-leftx righty-lefty],'LineWidth',boundingboxWidth,'Color','white');
    
    img_silhouetteSP = uint8(img_silhouetteSP);
    img_silhouetteSP = insertShape(img_silhouetteSP,'Rectangle',[leftx lefty rightx-leftx righty-lefty],'LineWidth',boundingboxWidth,'Color','white');    
    
    screen1 = fcn_DrawCross(ScreenWidth, ScreenHeight, ScreenHeight/2, ScreenWidth/2);
    %imshow(screen1);

    screen2 = fcn_DrawCross(ScreenWidth, ScreenHeight, ctrx, ctry);
    screen2 = uint8(screen2);
    screen2 = insertShape(screen2,'Rectangle',[leftx lefty rightx-leftx righty-lefty],'LineWidth',2,'Color','black');
    %screen2 = rgb2gray(screen2);   
    %imshow(screen2);

    %% Write GIF
    repeat = 0; %play only once for 0 and Inf for infinite looping
    screen3 = uint8(256/2*ones(ScreenWidth, ScreenHeight,3));
    
    namegif = ['gif_' num2str(t) '_' num2str(VHhumanStats(t).classid) ...
        '_' num2str(VHhumanStats(t).whereapt) '_' num2str(VHhumanStats(t).roomid) '.gif'];
    
    fcn_WriteGIF(screen1, screen2, img_silhouette, screen3, exptime, namegif,cf,[WriteDir keyframetype], repeat);
    imwrite(img_silhouette,[WriteDir keyframetype '/trial_oriimg_' num2str(t) '_' num2str(VHhumanStats(t).classid) ...
        '_' num2str(VHhumanStats(t).whereapt) '_' num2str(VHhumanStats(t).roomid) '.jpg']);
    
    
    %% Write GIF
    namegif = ['gif_' num2str(t) '_' num2str(VHhumanStats(t).classid) ...
        '_' num2str(VHhumanStats(t).whereapt) '_' num2str(VHhumanStats(t).roomid) '_sp.gif'];
    
    fcn_WriteGIF(screen1, screen2, img_silhouetteSP, screen3, exptime, namegif,cf,[WriteDir keyframetype], repeat);
    imwrite(img_silhouette,[WriteDir keyframetype '/trial_oriimg_' num2str(t) '_' num2str(VHhumanStats(t).classid) ...
        '_' num2str(VHhumanStats(t).whereapt) '_' num2str(VHhumanStats(t).roomid) '_sp.jpg']);
    
end












































