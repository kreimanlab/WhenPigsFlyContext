clear all; close all; clc;
load(['Mat/UnrelStats.mat']);
load(['Mat/DummyResponseGT.mat']);

headerimg = 'http://kreimanlab.com/mengmiMturkHost/Unrel/keyframe_UNREL/';

VHhumanStats = UnrelStats;
imagelist = extractfield(VHhumanStats,'mturkGTname')';
classnamelist = extractfield(VHhumanStats, 'label');
load(['Mat/mturk_GTlabel_unrel.mat']);

displayfigure = 0;
[meanRT, stdRT] = fcn_getReactionTimeBound(mturkData, displayfigure);

AllImgPresentedid = [];
Allresponse = [];

for i = 1:length(mturkData)
    ans = mturkData(i).answer; 
    dummyans = mturkData(i).dummy;
    if length(ans)<30 || length(dummyans) <3
        mturkData(i).subjvalid = 0;
        continue;
    end
    
    %condition 1: check dummy condition; if failed, this is bad subject
    CorrectNumDummy = fcn_checkDummyResponse(dummyans, DummyResponseGT);
    if CorrectNumDummy < 2
        mturkData(i).subjvalid = 0;
        continue;
    end 
    
    %condition 2: check reaction time below 1 std
    reactiontimelist = extractfield(ans,'rt')/1000; % in secs instead of millisecs
    selectedind = find(reactiontimelist < meanRT+stdRT);
    if length(selectedind) < 15 %below half of the trials
        mturkData(i).subjvalid = 0;
        continue;
    end
    
    mturkData(i).subjvalid = 1;
    responseList = extractfield(ans,'response');
    imgpresentid = extractfield(ans,'imageID');
    responseList = responseList(selectedind);
    imgpresentid = imgpresentid(selectedind);
    
    imgpresentid_clean = cellfun(@(x) x(length(headerimg)+1:end), imgpresentid, 'un', 0);
    responseList = lower(responseList);
    Allresponse = [Allresponse string(responseList)];
    AllImgPresentedid = [AllImgPresentedid string(imgpresentid_clean)];
end

%condition 3: sort descending in terms of number of occurences 
for i = 1:length(VHhumanStats)
    imgname = imagelist{i};
    response = Allresponse(find(strcmp(AllImgPresentedid,imgname)));     
    [GC,GR] = groupcounts(response');           
    VHhumanStats(i).RawGroupCounts = GC;
    VHhumanStats(i).RawGroups = GR;    
end

for i = 1:length(mturkData)
    ans = mturkData(i).answer; 
    if mturkData(i).subjvalid == 0
        continue;
    end
    responseList = extractfield(ans,'response');
    imgpresentid = extractfield(ans,'imageID');
    imgpresentid_clean = cellfun(@(x) x(length(headerimg)+1:end), imgpresentid, 'un', 0);
    responseList = lower(responseList);
    imgpresentid_clean = string(imgpresentid_clean);
    responseList = string(responseList);
    
    countNo = 0; %if more than 14 responses are always far off from majority; bad subj
    for j = 1:length(responseList)        
        rawgroupcount = VHhumanStats(find(strcmp(imgpresentid_clean(j),string(imagelist)))).RawGroupCounts;
        rawgroups = VHhumanStats(find(strcmp(imgpresentid_clean(j),string(imagelist)))).RawGroups;
        normGC = rawgroupcount/sum(rawgroupcount);
        score = normGC(find(strcmp(responseList(j), rawgroups)));
        if score <= 0.3
            countNo = countNo + 1;
        end
        if countNo > 15
            mturkData(i).subjvalid = 0;
            %warning(['lousy subj']);
            break;
        end
    end
end

save(['Mat/mturk_GTlabel_unrel_compiled.mat'], 'mturkData');

%% final steps to collect all response from good subjs as ground truths
AllImgPresentedid = [];
Allresponse = [];
goodSubj = 0;
for i = 1:length(mturkData)
    ans = mturkData(i).answer; 
    if mturkData(i).subjvalid == 0
        continue;
    end
    goodSubj = goodSubj + 1;
    responseList = extractfield(ans,'response');
    imgpresentid = extractfield(ans,'imageID');
    reactiontimelist = extractfield(ans,'rt')/1000; % in secs instead of millisecs
    selectedind = find(reactiontimelist < meanRT+stdRT);
    responseList = responseList(selectedind);
    imgpresentid = imgpresentid(selectedind);
    
    imgpresentid_clean = cellfun(@(x) x(length(headerimg)+1:end), imgpresentid, 'un', 0);
    responseList = lower(responseList);
    Allresponse = [Allresponse string(responseList)];
    AllImgPresentedid = [AllImgPresentedid string(imgpresentid_clean)];
end


for i = 1:length(VHhumanStats)
    imgname = imagelist{i};
    response = Allresponse(find(strcmp(AllImgPresentedid,imgname)));
    
    %condition 4: remove those dummy responses
    response(strcmp(response, "na")) = [];
    response(strcmp(response, "none")) = [];
    response(strcmp(response, "nothing")) = [];
    response(strcmp(response, "idontknow")) = [];
    response(strcmp(response, "unknown")) = [];
    response(strcmp(response, "error")) = [];
        
    VHhumanStats(i).HumanGT = response;
    [GC,GR] = groupcounts(response');           
    VHhumanStats(i).ProcessedResponseCounts = GC;
    VHhumanStats(i).ProcessedResponse = GR;
    
end

%% visualize some examples (bad ones)
% bad = [31,49,104,146,194,318,341,731];
% for i = bad
%     imgname = imagelist{i};
%     img = imread(strcat('/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/', imgname));
%     res = VHhumanStats(i).HumanGT;
%     uniall = unique(res);
%     uniall = strjoin(uniall,' ,');
%     ans = [classnamelist{i} '(' num2str(length(res)) ') : ' uniall ];
%     ans = strjoin(ans);
%     imshow(img);
%     title(ans{1});
%     pause;
% end


%% collocate responses for each 36 categories
ClassOnlyList = unique(classnamelist);

for i = 1:length(ClassOnlyList)
    ind = find(strcmp(classnamelist, ClassOnlyList(i)));
    allresponse = extractfield(VHhumanStats(ind),'HumanGT');
    all = [];
    for j = 1:length(allresponse)
        all = [all allresponse{j}];
    end
    [GC,GR] = groupcounts(all');
    [a b] = sort(GC,'descend');
    GC = GC(b);
    GR = GR(b);
    for j = ind
        VHhumanStats(j).ProcessedClassGT = GR;
        VHhumanStats(j).ProcessedClassGTCounts = GC;
    end
    printGC = string(GC);
    printall = [GR printGC];
    printall = join(printall,'-');
    printall = printall';
    printall = join(printall, ',');
    ans = [ClassOnlyList{i} '(' num2str(length(all)) ') : ' printall ];
    ans = strjoin(ans);
    display(ans{1});
end
    
UnrelStats = VHhumanStats;
save(['Mat/UnrelStats_GTresult_unrel.mat'], 'UnrelStats');
    
    