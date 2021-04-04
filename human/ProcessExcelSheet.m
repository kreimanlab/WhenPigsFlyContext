clear all; close all; clc;

ItemInRoomTable = readtable('raw/items_in_rooms_clean.xlsx');
SurfaceRoomTable = readtable('raw/surfaces_in_rooms_clean.xlsx');

ItemToRoom = ItemInRoomTable{:,2:5};

wantedClass = ItemInRoomTable{:,1}; 
wantedClass = char(wantedClass);
save(['raw/ItemToRoom.mat'],'ItemToRoom','wantedClass');

SurfaceToRoom = SurfaceRoomTable{:,2:5};
SurfaceList = SurfaceRoomTable{:,1};
SurfaceList = char(SurfaceList); % 3*20 char array
% Save A to mat file (format is not HDF5).
save(['raw/SurfaceToRoom.mat'],'SurfaceToRoom','SurfaceList');

%%  for abnormal location
% load(['Mat/VHhumanStats.mat']);
% badind = [];
% for i = 1:length(wantedClass)
%     
%     wanted = wantedClass(i,:);
%     wanted= wanted(~isspace(wanted));
%     if length(find(strcmp(wanted, filtered_class))) < 1
%         badind = [badind i];
%     end   
% end
% 
% ItemToRoom(badind, :) = [];
% wantedClass(badind,:) = [];
% ItemToRoom = 1-ItemToRoom;

subjname = {'gk','dimitar','phillip'};
all =zeros(36,20);
for subj = 1:length(subjname)
    
    ItemInRoomTable = readtable(['raw/items_in_locations_anomaly_' subjname{subj} '.xlsx']);    
    ItemToRoom = ItemInRoomTable{:,2:end};
    ItemToRoom(isnan(ItemToRoom)) = 0;
    all = all + ItemToRoom;
end

ItemToRoom = int32(all>=2);
wantedClass = ItemInRoomTable{:,1}; 
wantedClass = char(wantedClass);

SurfaceRoomTable = readtable('raw/surfaces_in_rooms_anomaly.xlsx');
SurfaceList = SurfaceRoomTable{:,1};
SurfaceList = char(SurfaceList); % 3*20 char array

save(['raw/ItemToRoom_anomaly.mat'],'ItemToRoom','wantedClass','SurfaceList');











