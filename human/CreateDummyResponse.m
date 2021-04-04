clear all; close all; clc;

DummyTotal = 6;
load('Mat/mturk_GTlabel.mat');
Response = cell(1,DummyTotal);

for i = 1:length(mturkData)
    subjdummy = mturkData(i).dummy;
    if length(subjdummy) < 2
        continue;
    end
    responselist = extractfield(subjdummy, 'response');
    dummyidlist = extractfield(subjdummy, 'imageID');
    dummyidlist = cellfun(@(x) str2num(x(end-4)), dummyidlist);
    
    for j = 1:length(dummyidlist)
        Response{ dummyidlist(j) } = [Response{ dummyidlist(j) } string(responselist{j}) ];
    end
    
end

FinalResponseList = cell(1,DummyTotal);
for i = 1:DummyTotal
    res = lower(Response{i});
    splitted = unique(res);
    FinalResponseList{i} = splitted;
end

%%add hoc check for dummy ground truth
content = FinalResponseList{2};
FinalResponseList{2} = content(1:10);
content = FinalResponseList{3};
content([3,14,15,22]) = [];
FinalResponseList{3} = content;
content = FinalResponseList{4};
content([3:7]) = [];
FinalResponseList{4} = content;

DummyResponseGT = FinalResponseList;
save('Mat/DummyResponseGT.mat','DummyResponseGT');
