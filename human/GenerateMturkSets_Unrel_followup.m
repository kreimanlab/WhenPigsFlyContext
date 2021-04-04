clear all; close all; clc;

% load(['Mat/mturk_expUnrel_results.mat']);
% newlist = [];
% Limit = 10;
% for i = 1:length(UnrelStats)
%     if isempty(UnrelStats(i).GIFresponse)
%         continue;
%     end
%     
%     num = length(UnrelStats(i).HumanGT);
%     remaining = Limit - num;
%     for j = 1:remaining
%         replicate = UnrelStats(i);
%         newlist = [newlist replicate];
%     end
% end
% save('temp.mat','newlist');

load('temp.mat');
UnrelStats = newlist;
%NumClass = length(classnamelist);
NumSets = 2500;
storeDir = 'MturkSets/exp_Unrel_followup/';
mkdir(storeDir);

classlist = extractfield(UnrelStats,'labelid');
imgidlist = extractfield(UnrelStats,'imgid');
objidlist = extractfield(UnrelStats,'objid');

[GC,GR] = groupcounts(classlist');
[t,ind] = sort(GC, 'descend');
GR = GR(ind);
resort_classlist =[];
for i = GR'
    ind = find(classlist == i);
    ind = ind(randperm(length(ind)));
    resort_classlist = [resort_classlist ind];
end
classlist = classlist(resort_classlist);
imgidlist = imgidlist(resort_classlist);
objidlist = objidlist(resort_classlist);

NumTrials = 4;
classlistF = [classlist classlist(end-1082:end)];
imgidlistF = [imgidlist imgidlist(end-1082:end)];
objidlistF = [objidlist objidlist(end-1082:end)];

classlistF = reshape(classlistF, [t(1) 4]);
imgidlistF = reshape(imgidlistF, [t(1) 4]);
objidlistF = reshape(objidlistF, [t(1) 4]);

for n = 1:size(classlistF,1)   
    
    display(['processing set: ' num2str(n)]);   
    
    %write to text files for mturk exp
    filePh = fopen([storeDir 'mturkSet_' num2str(n) '.txt'],'w');
    comb = [imgidlistF(n,:); objidlistF(n,:)];
    comb = comb(:);
    fprintf(filePh,'%d\n',comb);
    fclose(filePh);         
end

%% sanity check

load(['Mat/mturk_expUnrel_results.mat']);
classlistF = classlistF(:);
imgidlistF = imgidlistF(:);
objidlistF = objidlistF(:);

for i = 1:length(UnrelStats)
    imgid = UnrelStats(i).imgid;
    labelid = UnrelStats(i).labelid;
    objid = UnrelStats(i).objid;
    
    UnrelStats(i).tempcount = length(find(classlistF == labelid & imgidlistF == imgid & objidlistF == objid));
end



