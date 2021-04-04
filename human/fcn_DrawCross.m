function [screen] = fcn_DrawCross(screenwidth, screenheight, ctrx, ctry)
%FCN_DRAWCROSS Summary of this function goes here
%   Detailed explanation goes here
    w = screenwidth; h=screenheight;
    crosswidth = 2; crossheight = 28;
    screen = ones(w,h,3)*128;
    screen(ctry-crosswidth/2:ctry+crosswidth/2, ctrx-crossheight/2:ctrx+crossheight/2,:) = 0;
    screen(ctry-crossheight/2:ctry+crossheight/2, ctrx-crosswidth/2:ctrx+crosswidth/2,:)=0;
    screen = uint8(screen);
    %imglocation = ['temp/fixationscreen.jpg'];
    %imwrite(screen,imglocation);
    %imshow(screen)    
end

