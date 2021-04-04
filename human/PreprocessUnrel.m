clear all; close all; clc;

Testset = readtable('/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/matlab/unrel-dataset/Philipp_UnRel_Testset.csv');
%pre-process testset and convert to structure
UnrelStats = [];
NumImg = size(Testset,1);
classnamelist = unique(Testset{:,6});
for t = 1:NumImg
    infor.imgname = Testset{t,1}{1};
    infor.imgid = Testset{t,2};
    infor.imgwidth = Testset{t,3};
    infor.imgheight = Testset{t,4};
    infor.objid = Testset{t,5};
    infor.label = Testset{t,6}{1};
    infor.labelid = find(strcmp(Testset{t,6}{1}, classnamelist));
    infor.bbox_x = Testset{t,7};
    infor.bbox_y = Testset{t,8};
    infor.bbox_width = Testset{t,9};
    infor.bbox_height = Testset{t,10};
    infor.mturkGTname = ['trial_oriimg_' num2str(Testset{t,2}) '_' num2str(Testset{t,5}) '.jpg'];
    infor.mturkGIFname = ['gif_' num2str(Testset{t,2}) '_' num2str(Testset{t,5}) '.gif'];
    UnrelStats = [UnrelStats infor];
end
save(['Mat/UnrelStats.mat'],'UnrelStats','classnamelist');

ImgDir = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/matlab/unrel-dataset/images/';

ScreenWidth = 1024;
ScreenHeight = 1280;
exptime = [500 1000 200 100]; %in millisecs
cf = 100; %this is the greatest common factor of experiment time
boundingboxWidth = 3;
keyframetype = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/matlab/Stimulus_unrel/keyframe_UNREL'; %change this to keyframe_expB, etc

%clean and create two folders for the type of expeirment
%rmdir([keyframetype], 's');
%rmdir([keyframetype '_gif'], 's');
mkdir([keyframetype]);
mkdir([keyframetype '_gif']);

for t = 1:NumImg
    imgname = Testset{t,1}{1};
    img = imread([ImgDir imgname]);
    img = imresize(img, [Testset{t,4}, Testset{t,3}]);
    
    if length(size(img))~=3
        img = cat(3, img, img, img);
    end
    
    bboxpos = [Testset{t,7} Testset{t,8} Testset{t,9} Testset{t,10}];
    imgbbox = insertShape(img,'Rectangle',bboxpos,'LineWidth',boundingboxWidth,'Color','red');
    
    bin = zeros([size(img,1) size(img,2)]);
    bin(bboxpos(2):bboxpos(2)+bboxpos(4), bboxpos(1):bboxpos(1)+bboxpos(3)) = 255;
    
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
    
    imwrite(binscreen,[keyframetype '/trial_binimg_' num2str(Testset{t,2}) '_' num2str(Testset{t,5}) '.jpg']);
    
    % extract bounding box; x is horinzontal axis; y is vertical axis; (0,0) is
    % at top left corner of the image
    [row, col] = find(binscreen(:,:,1)==1);
    leftx = min(col);lefty = min(row);rightx = max(col);righty = max(row);
    ctrx = floor((leftx + rightx)/2);ctry = floor((lefty + righty)/2);
    oh = rightx - leftx;
    ow = righty - lefty; 
    
    img_complete = screen; %ratio = 1
    img_complete = uint8(img_complete);
    
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
    
    namegif = ['gif_' num2str(Testset{t,2}) '_' num2str(Testset{t,5}) '.gif'];
    fcn_WriteGIF(screen1, screen2, img_complete, screen3, exptime, namegif,cf,keyframetype, repeat);
    imwrite(img_complete,[keyframetype '/trial_oriimg_' num2str(Testset{t,2}) '_' num2str(Testset{t,5}) '.jpg']);
    
end












































