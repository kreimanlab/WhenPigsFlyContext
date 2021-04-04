clear all; close all; clc;
load(['Mat/UnrelStats_GTresult_unrel.mat']);
VHhumanStats = UnrelStats;
imgidlist = extractfield(UnrelStats, 'imgid');
objidlist = extractfield(UnrelStats, 'objid');
classidlist = extractfield(UnrelStats, 'labelid');

expname = 'expUnrel';
%match humans to gt labels
load(['Mat/mturk_expUnrel.mat']);

classuniqueidlist = unique(classidlist);
TotalNumImg = 29;
MaxSubj = length(mturkData);
classlabelSet = 2;
TotalTrialNumSet = 33;

%store infor about cate, type
startingMturkData = 1;

for i = startingMturkData:length(mturkData)
    ans = mturkData(i).answer;
    %check if the subject has not provided more than 10 answers; discard
    %subjects
    if length(ans) <TotalNumImg
        continue;
    end
    
    responseList = extractfield(ans,'response');
    responseList = lower(responseList);
    
    %check words with 1 or 2 letters for a word more than 10 times; discard
    %subjects
    lenList = cellfun(@(x){length(x)},responseList);
    lenList = cell2mat(lenList);
    if length(find(lenList<3))>5
        continue;
    end
    
    %check repetitions of single words more than 10 times; discard
    %subjects
    forbidden = unique(responseList);
    counter = 0;
    for f= 1:length(forbidden)
        if length(find(strcmp(forbidden{f},responseList)))>5
            counter = 1;
            break;
        end
    end    
    if counter>0
        continue;
    end
    
    %check forbidden word repeating more than 5 times; discard
    %subjects
    forbidden = {'toosmall','Toosmall','glitched','unsure','unknown',' don''t','no','na','none','idk','dontknow','bullshit','clueless','nothing','sth','error'};
    counter = 0;
    for f= 1:length(forbidden)
        counter = counter + length(find(strcmp(forbidden{f},responseList)));
    end    
    if counter>5
        continue;
    end
        
    type = ans(1).counterbalance;
    
    %infor = [i binL(i) cateL(i) imgL(i) typeL(i)];
    %load(['/home/mengmi/Projects/Proj_context2/mturk/Mturk/StimulusBackUp/expA_what/mturk_set' num2str(type) '/infor.mat']);
    infor = importdata(['/home/mengmi/Projects/Proj_context3/VirtualHome/matlab/MturkSets/exp_Unrel/mturkSet_' num2str(type) '.txt']);
    infor = reshape(infor, classlabelSet, TotalTrialNumSet)';
    for j = 1:length(ans)
        vec = infor(ans(j).hit,:);
        indimg = find(imgidlist == vec(1) & objidlist == vec(2));
        gt  = VHhumanStats(indimg).ProcessedClassGT;        
        res = responseList{j};
        
        flag = 0;
        if length(res)<3
            correct = nan;
            flag = 1;
        end
        counter = 0;
        forbidden = {' don''t','no','none','idk','dontknow','bullshit','clueless','nothing','sth','unknown','error'};
        for f= 1:length(forbidden)
            if strcmp(forbidden{f},res)
                counter = 1;
                break;
            end
        end 
        if counter == 1
            correct = nan;
            flag = 1;
        end
        
        if flag == 0
            if fcn_spellcheck(res, gt)==1 || length(find(strcmp(res, gt)==1)) > 0
                correct = 1;
            else                
                correct = 0;
            end
        end
          
        mturkData(i).answer(j).gt = gt;
        mturkData(i).answer(j).correct = correct;
        mturkData(i).answer(j).objid = vec(2);
        mturkData(i).answer(j).imgid = vec(1);
        mturkData(i).answer(j).labelid = VHhumanStats(vec(2)).labelid;
    end
end


save(['Mat/mturk_' expname '_compiled.mat'],'mturkData');

%% convert correctness into UnrelStats 
HResponselist = [];
HCorrectlist = [];
Hobjidlist = [];
Hlabelidlist = [];

for i = 1:length(mturkData)
    
    if ~isfield(mturkData(i).answer,'correct')
        continue;
    end

    HResponselist = [HResponselist extractfield(mturkData(i).answer,'response')];
    HCorrectlist = [HCorrectlist extractfield(mturkData(i).answer,'correct')];
    Hobjidlist =[Hobjidlist extractfield(mturkData(i).answer,'objid')];
    Hlabelidlist = [Hlabelidlist extractfield(mturkData(i).answer,'labelid')];
end

nanmean(HCorrectlist)
HResponselist = string(HResponselist);

for i = 1:length(VHhumanStats)
    objid = VHhumanStats(i).objid;
    indselected = find(Hobjidlist == objid);
    VHhumanStats(i).GIFresponse = HResponselist(indselected);
    VHhumanStats(i).GIFcorrect = HCorrectlist(indselected);
end
UnrelStats = VHhumanStats;

save(['Mat/mturk_' expname '_results.mat'],'UnrelStats');
%display accuracy of humans based on class
overall = [];
for i = 1:length(classuniqueidlist)
    labelid = classuniqueidlist(i);
    selectedind = find(classidlist == labelid);
    label = VHhumanStats(selectedind(1)).label;
    meanaccu = nanmean(HCorrectlist(find(Hlabelidlist == labelid)));
    overall = [overall meanaccu];
    display(['Accuracy (' label ') = ' num2str(meanaccu)]);
end
display(['overall = ' num2str(nanmean(overall)) ';rmse = ' num2str(nanstd(overall)/sqrt(length(overall)))]);






