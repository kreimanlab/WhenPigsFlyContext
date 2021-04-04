clear all; close all; clc;

for apartment = [6]
    
    fid = fopen(['/home/mengmi/Desktop/stats/apartment_' num2str(apartment) '_good.txt']);
    tline = fgetl(fid);
    while ischar(tline)
        disp(tline);
        imshow(imread(['../gitsource/demo/stimulus/' tline]));
        title(['apartment ' num2str(apartment) '; ' tline]);
        pause;
        tline = fgetl(fid);
    end
    fclose(fid);
    
end