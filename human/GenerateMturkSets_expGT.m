clear all; close all; clc;

load(['Mat/VHhumanStats.mat']);
NumRooms = 4;
NumApt = 7;
NumSets = 800;
storeDir = 'MturkSets/exp_GT/';
mkdir(storeDir);

classlist = extractfield(VHhumanStats,'classid');
roomlist = extractfield(VHhumanStats,'roomid');
imagelist = extractfield(VHhumanStats,'imgname');
aptlist = extractfield(VHhumanStats,'whereapt');

for n = 1:NumSets    
    
    display(['processing set: ' num2str(n)]);   
    validflag = 0;
    
    while validflag == 0        
        classtrack = [];
        roomtrack = [];
        apttrack = [];
        imgtrack = {};        
        
        for a = randperm(NumApt)
            for r = randperm(NumRooms)
                indlist = find(roomlist == r & aptlist == a);
                classok = classlist(indlist);
                imageok = imagelist(indlist);

                %randomly pick one class
                randindlist = randperm(length(classok));
                counter = 1;
                while counter <= length(randindlist)
                    randind = randindlist(counter);
                    if ~any (classtrack == classok(randind))
                        apttrack=[apttrack a];
                        roomtrack = [roomtrack r];
                        classtrack = [classtrack classok(randind)];
                        imgtrack = [imgtrack imageok(randind)];
                        break;
                    end

                    counter = counter + 1;
                end

            end
        end

        if length(classtrack) >27 % NumApt *NumRooms
            validflag = 1;
            %write to text files for mturk exp
            filePh = fopen([storeDir 'mturkSet_' num2str(n) '.txt'],'w');
            fprintf(filePh,'%s\n',imgtrack{:});
            fclose(filePh);           
        end
    end
end

