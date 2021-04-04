clear all; close all; clc;
load(['Mat/VHhumanStats_GTresult.mat']);
VHGT = VHhumanStats;
classidlist = string(extractfield(VHGT,'classname'));
load(['Mat/VHhumanStats_size.mat']);
%copy ground truth responses collected to each class in VHgravity
for i = 1:length(VHhumanStats)
    indselected = find( strcmp(classidlist,VHhumanStats(i).classname));
    if length(indselected) > 0
        VHhumanStats(i).ProcessedClassGT = VHGT(indselected(1)).ProcessedClassGT;
        VHhumanStats(i).ProcessedClassGTCounts = VHGT(indselected(1)).ProcessedClassGTCounts;
    else
        error('we should not be here');
%         VHhumanStats(i).ProcessedClassGT = nan;
%         VHhumanStats(i).ProcessedClassGTCounts = 0;
    end
end

classidlist = extractfield(VHhumanStats, 'classid');

expname = 'expSize';
load(['Mat/mturk_expSize.mat']);

classuniqueidlist = unique(classidlist);
TotalNumImg = 20;
%classlabelSet = 2;

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
    infor = importdata(['/home/mengmi/Projects/Proj_context3/VirtualHome/matlab/MturkSets/exp_size/mturkSet_' num2str(type) '.txt']);
    %infor = reshape(infor, classlabelSet, TotalTrialNumSet)';
    for j = 1:length(ans)
        
        vec = infor(ans(j).hit);
        gifparts = split(vec, '_');
        indimg = str2num(gifparts{2});
        gt  = VHhumanStats(indimg).ProcessedClassGT; 
        %gtcheck = VHhumanStats(indimg).ProcessedClassGTCounts;
        res = responseList{j};
        
%         if gtcheck == 0
%             mturkData(i).answer(j).gt = gt;
%             mturkData(i).answer(j).correct = nan;
%             mturkData(i).answer(j).StatsID = indimg;
%             mturkData(i).answer(j).classid = VHhumanStats(indimg).classid;
%             mturkData(i).answer(j).whereapt = VHhumanStats(indimg).whereapt;
%             mturkData(i).answer(j).roomid = VHhumanStats(indimg).roomid;
%             mturkData(i).answer(j).classname = VHhumanStats(indimg).classname;
%             continue;
%         end
        
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
            if fcn_spellcheck(res, gt)==1  || length(find(strcmp(res, gt)==1)) > 0
            %if strcmp(res, gt)
                correct = 1;
            else
                correct = 0;
            end
        end
          
        mturkData(i).answer(j).gt = gt;
        mturkData(i).answer(j).correct = correct;
        mturkData(i).answer(j).StatsID = indimg;
        mturkData(i).answer(j).classid = VHhumanStats(indimg).classid;
        mturkData(i).answer(j).whereapt = VHhumanStats(indimg).whereapt;
        mturkData(i).answer(j).roomid = VHhumanStats(indimg).roomid;
        mturkData(i).answer(j).classname = VHhumanStats(indimg).classname;
        mturkData(i).answer(j).condSize = VHhumanStats(indimg).condSize;
    end
end


save(['Mat/mturk_' expname '_compiled.mat'],'mturkData');

%% convert correctness into UnrelStats 
HResponselist = [];
HCorrectlist = [];
Hobjidlist = [];
Hlabelidlist = [];
HcondWalllist = [];

for i = 1:length(mturkData)
    
    if ~isfield(mturkData(i).answer,'correct')
        continue;
    end

    HResponselist = [HResponselist extractfield(mturkData(i).answer,'response')];
    HCorrectlist = [HCorrectlist extractfield(mturkData(i).answer,'correct')];
    Hobjidlist =[Hobjidlist extractfield(mturkData(i).answer,'StatsID')];
    Hlabelidlist = [Hlabelidlist extractfield(mturkData(i).answer,'classid')];
    HcondWalllist = [HcondWalllist extractfield(mturkData(i).answer,'condSize')];
end

nanmean(HCorrectlist)
HResponselist = string(HResponselist);

for i = 1:length(VHhumanStats)
    indselected = find(Hobjidlist == i);
    VHhumanStats(i).GIFresponse = HResponselist(indselected);
    VHhumanStats(i).GIFcorrect = HCorrectlist(indselected);
    
    
end

%save(['Mat/mturk_' expname '_results.mat'],'VHhumanStats');
%display accuracy of humans based on class
overall = [];
total = 0;
for i = 1:length(classuniqueidlist)
    labelid = classuniqueidlist(i);
    selectedind = find(classidlist == labelid);
    
%     if VHhumanStats(selectedind(1)).ProcessedClassGTCounts == 0
%         continue;
%     end
    total = total + length(HCorrectlist(find(Hlabelidlist == labelid)));
    label = VHhumanStats(selectedind(1)).classname;
    meanaccu = nanmean(HCorrectlist(find(Hlabelidlist == labelid)));
    overall = [overall meanaccu];
    display(['Accuracy (' label ') = ' num2str(meanaccu)]);
end
total
display(['overall = ' num2str(nanmean(overall)) ';rmse = ' num2str(nanstd(overall)/sqrt(length(overall)))]);

%display accuracy of humans based on class
total = 0;
overall_S2 = [];
for i = 1:length(classuniqueidlist)
    labelid = classuniqueidlist(i);
    selectedind = find(classidlist == labelid);
    
%     if VHhumanStats(selectedind(1)).ProcessedClassGTCounts == 0
%         continue;
%     end
    
    label = VHhumanStats(selectedind(1)).classname;
    total = total + length(HCorrectlist(find(Hlabelidlist == labelid & HcondWalllist == 2)));
    meanaccu = nanmean(HCorrectlist(find(Hlabelidlist == labelid & HcondWalllist == 2)));
    overall_S2 = [overall_S2 meanaccu];
    display(['Accuracy (' label ') = ' num2str(meanaccu)]);
end
total
display(['overall(S2) = ' num2str(nanmean(overall_S2)) ';rmse = ' num2str(nanstd(overall_S2)/sqrt(length(overall_S2)))]);

%display accuracy of humans based on class
total = 0;
overall_S3 = [];
for i = 1:length(classuniqueidlist)
    labelid = classuniqueidlist(i);
    selectedind = find(classidlist == labelid);
    
%     if VHhumanStats(selectedind(1)).ProcessedClassGTCounts == 0
%         continue;
%     end
    
    label = VHhumanStats(selectedind(1)).classname;
    total = total + length(HCorrectlist(find(Hlabelidlist == labelid & HcondWalllist == 3 )));
    meanaccu = nanmean(HCorrectlist(find(Hlabelidlist == labelid & HcondWalllist == 3 )));
    overall_S3 = [overall_S3 meanaccu];
    display(['Accuracy (' label ') = ' num2str(meanaccu)]);
end
total
display(['overall(S3) = ' num2str(nanmean(overall_S3)) ';rmse = ' num2str(nanstd(overall_S3)/sqrt(length(overall_S3)))]);

%display accuracy of humans based on class
total = 0;
overall_S4 = [];
for i = 1:length(classuniqueidlist)
    labelid = classuniqueidlist(i);
    selectedind = find(classidlist == labelid);
    total = total + length(HCorrectlist(find(Hlabelidlist == labelid & HcondWalllist == 4 )));
%     if VHhumanStats(selectedind(1)).ProcessedClassGTCounts == 0
%         continue;
%     end
    
    label = VHhumanStats(selectedind(1)).classname;
    meanaccu = nanmean(HCorrectlist(find(Hlabelidlist == labelid & HcondWalllist == 4 )));
    overall_S4 = [overall_S4 meanaccu];
    display(['Accuracy (' label ') = ' num2str(meanaccu)]);
end
total
display(['overall(S4) = ' num2str(nanmean(overall_S4)) ';rmse = ' num2str(nanstd(overall_S4)/sqrt(length(overall_S4)))]);


save(['Mat/mturk_' expname '_results.mat'],'VHhumanStats','overall','overall_S2','overall_S3','overall_S4');
