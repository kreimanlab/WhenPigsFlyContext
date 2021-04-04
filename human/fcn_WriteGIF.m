function fcn_WriteGIF( screen1, screen2, img, screen3, exptime, namegif,cf, keyframetype, repeattime)
%FCN_WRITEGIF Summary of this function goes here
%   Detailed explanation goes here

    keyframe ={};
    
    %rgb = cat(3, screen1, screen1, screen1);
    rgb = screen1;
    keyframe = [keyframe rgb]; 
    %rgb = cat(3, screen2, screen2, screen2);
    rgb = screen2;
    keyframe = [keyframe rgb];
    %rgb = cat(3, img, img, img);
    rgb = img;
    keyframe = [keyframe rgb]; 
    %rgb = cat(3, screen3, screen3, screen3);
    rgb = screen3;
    keyframe = [keyframe rgb];
    
    keyloopnum = exptime/cf;

    counter = 0;
    
    display(['writing ... ']);
    for i = 1:length(exptime)
        
        for j = 1:keyloopnum(i)
            counter = counter + 1;            
            [imind,cm] = rgb2ind(keyframe{i},256);           
            if counter == 1 
              imwrite(imind,cm,[keyframetype '_gif/' namegif],'gif','DelayTime',cf/1000, 'Loopcount',repeattime); 
            else 
              imwrite(imind,cm,[keyframetype '_gif/' namegif],'gif','DelayTime',cf/1000,'WriteMode','append'); 
            end
            
        end     
    end
 
    display(['saved to ' keyframetype '_gif/' namegif]);

%end

