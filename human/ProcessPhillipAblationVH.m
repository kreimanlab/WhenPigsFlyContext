clear all; close all; clc;

baselinelist = {'A1_shareEncoder','A2_targetonly','A3_contextonly','A4_nodettach'};
condlist = {'no_context','gravity','coocurrence','coocurrence_and_gravity','size','IC'};
mapmatind = [4,2,3,3,5,1];
expnamelist = {'expNormal', 'expGravity', 'expAnomaly', 'expNoContext', 'expSize'}; %,'expMaterial

%% Setup the Import Options
opts = delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["imagename", "groundtruth", "predicted", "correctness"];
opts.VariableTypes = ["string", "categorical", "categorical", "double"];
opts = setvaropts(opts, 1, "WhitespaceRule", "preserve");
opts = setvaropts(opts, [1, 2, 3], "EmptyFieldRule", "auto");
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

NumObjSizes = 2;
NumConds = length(condlist);
modelresult = nan(length(baselinelist), NumObjSizes, NumConds);
modelresult_std = nan(length(baselinelist), NumObjSizes, NumConds);

% Import the data
for b = [1:length(baselinelist)]
    for selected = [1:length(condlist)]
        
        Pjason = readtable(['/home/mengmi/Projects/Proj_context3/VirtualHome/matlab/Mat/Ajason_' baselinelist{b} '_' condlist{selected} '.csv'], opts);
        load(['Mat/mturk_' expnamelist{mapmatind(selected)} '_results.mat']);
        
        if selected == 1
            jasonimgname = Pjason{:,1};
            jasoncorrect = Pjason{:,4};
            newStr = split(jasonimgname,'_');
            newStr = newStr(:,3);
            newStr = str2double(newStr);            
            CORatio = extractfield(VHhumanStats(newStr),'dva');                    
        else
            jasonimgname = Pjason{:,1};
            jasoncorrect = Pjason{:,4};
            GTimgnamelist = extractfield(VHhumanStats, 'imgname');
            [ind imgmatch] = ismember(jasonimgname, string(GTimgnamelist));
            CORatio = extractfield(VHhumanStats(imgmatch),'dva');            
        end       
        
        modelresult(b,1,selected) = nanmean(jasoncorrect(find(CORatio<=2)));
        modelresult(b,2,selected) = nanmean(jasoncorrect(find(CORatio>2))); 
        
        temp = jasoncorrect(find(CORatio<=2));
        modelresult_std(b,1,selected) = nanstd(temp)/sqrt(length(temp));
        temp = jasoncorrect(find(CORatio>2));
        modelresult_std(b,2,selected) = nanstd(temp)/sqrt(length(temp));
        
    end
end

save(['Mat/ablationresult.mat'],'modelresult','modelresult_std');

