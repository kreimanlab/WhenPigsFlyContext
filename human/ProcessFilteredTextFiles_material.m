clc; clear all; close all;

NumApts = 7;
NumRooms = 4;
NumThres = 7; %must be in all 7 rooms
load('Mat/VHhumanStats.mat');
PixToVA = 156/5; %pixels per visual angle in degrees

%this mat is manually imported into matlab "jason_material_combined.csv"
T = load('Mat/jasonmaterialcombined.mat');
T= T.jasonmaterialcombined;
classlist = string(T{1:end,1});
apartmentlist = T{1:end,2};
roomlist = string(T{1:end,3});
imagelist = string(T{1:end,4});
leftlist = T{1:end,5};
%leftlist = cellfun(@str2num,leftlist);

toplist = T{1:end,6};
%toplist = cellfun(@str2num,toplist);

rightlist = T{1:end,7};
%rightlist = cellfun(@str2num,rightlist);

bottomlist = T{1:end,8};
%bottomlist = cellfun(@str2num,bottomlist);


classAll = unique(classlist);
display(length(classAll));

%% check whether the filtered classes are in every apartment
% for classid = 1:length(filtered_class)
%     
%     classname = filtered_class{classid};
%     %display(classname);
%     whereapt = unique(apartmentlist(find(classlist == classname)));
%     if length(whereapt) < NumThres
%         display(classname);
%     end
%     
% end
%filtered_class = extractfield(VHhumanStats, 'classname');
countAll = zeros(length(filtered_class), NumApts, NumRooms);
VHhumanStats = [];

for classid = 1:length(filtered_class) 
    
    classname = filtered_class{classid};
    
    whereimg = imagelist(find(classlist == classname));
    whereapt = apartmentlist(find(classlist == classname)) + 1;
%     if length(unique(whereapt)) < NumThres
%         error(['we should not be here']);
%     end
    
    whereroom = roomlist(find(classlist == classname));
    whereleft = leftlist(find(classlist == classname));
    whereright = rightlist(find(classlist == classname));
    wheretop = toplist(find(classlist == classname));
    wherebottom = bottomlist(find(classlist == classname));
    
    for i= 1:length(whereapt)
        ind_apt = whereapt(i);
        [ind_room temp] = find(whereroom(i) == UniqueRoomList);
        countAll(classid, ind_apt, ind_room) = countAll(classid, ind_apt, ind_room) + 1;
        view = squeeze(countAll(classid,:,:));
        imgname = whereimg(i);
        
        infor.classname = classname;
        infor.classid = classid;
        infor.whereapt = ind_apt;
        infor.whereroom = whereroom(i);
        infor.roomid = ind_room;
        infor.imgname = strtrim(imgname);
        infor.left = whereleft(i);
        infor.right = whereright(i);
        infor.top = wheretop(i);
        infor.bottom = wherebottom(i);
        VHhumanStats = [VHhumanStats infor];
        
    end
    
    %view = squeeze(countAll(classid,:,:));
    display(view);
end

save('Mat/VHhumanStats_material.mat','VHhumanStats','countAll','UniqueRoomList','filtered_class');


