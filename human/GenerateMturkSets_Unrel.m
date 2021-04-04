clear all; close all; clc;

load(['Mat/UnrelStats.mat']);
NumClass = length(classnamelist);
NumSets = 800;
storeDir = 'MturkSets/exp_Unrel/';
mkdir(storeDir);

classlist = extractfield(UnrelStats,'labelid');
imgidlist = extractfield(UnrelStats,'imgid');
objidlist = extractfield(UnrelStats,'objid');

for n = 1:NumSets    
    
    display(['processing set: ' num2str(n)]);   
    validflag = 0;
    
    while validflag == 0        
        classidtrack = [];
        imgidtrack = [];
        imgidselected = [];    
        objidselected = [];
        
        for a = randperm(length(classnamelist))
            
                indlist = find(classlist == a);
                imgidok = imgidlist(indlist);
                objidok = objidlist(indlist);

                %randomly pick one class
                randindlist = randperm(length(imgidok));
                counter = 1;
                while counter <= length(randindlist)
                    randind = randindlist(counter);
                    if ~any (imgidtrack == imgidok(randind))
                        
                        classidtrack = [classidtrack a];
                        imgidtrack =[imgidtrack imgidok(randind)];
                        imgidselected = [imgidselected imgidok(randind)];
                        objidselected = [objidselected objidok(randind)];
                        break;
                    end

                    counter = counter + 1;
                end

        end
        
        if length(imgidtrack) >= length(classnamelist) % NumApt *NumRooms
            validflag = 1;
            %write to text files for mturk exp
            filePh = fopen([storeDir 'mturkSet_' num2str(n) '.txt'],'w');
            comb = [imgidselected; objidselected];
            comb = comb(:);
            fprintf(filePh,'%d\n',comb);
            fclose(filePh);           
        end            
        
    end

        
end

