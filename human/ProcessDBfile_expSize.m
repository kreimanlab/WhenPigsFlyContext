clear all; close all; clc;

csvname = 'csv/results_size.csv';
system(['sqlite3 -header -csv db/expSize.db "select * from expSize;" > ' csvname]);

fid = fopen(csvname);
out = textscan(fid,'%s');
strtotal = out{1,1};
pattern = 'http://kreimanlab.com/mengmiMturkHost/VirtualHome/keyframe_VH_size_gif/';

savefilename = 'Mat/mturk_expSize.mat';

mturkData = [];
answer = [];  
for i = 1:length(strtotal)

    str = strtotal{i,1};    
    k = strfind(str,pattern);    
    if isempty(k)
        continue;
    end
    
    if isempty(strfind(strtotal{i-12},'""uniqueid"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i-8},'{""current_trial"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i-5},'{""rt"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i-3},'""hit"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i-1},'""imageID"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i+1},'""trial"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i+3},'""counterbalance"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i+7},'""response"":'))
        continue;
    end
    
    strpart = strtotal{i-11};
    strpart = strsplit(strpart,':');
    workerid = strpart{1}(3:end);
    assignmentid = strpart{2}(1:end-3);
    imageID = strtotal{i}(3:end-3);
    %remove %20 from the string if any
    if strfind(imageID, '%20')
        imageID = strrep(imageID,'%20',' ');
    end    
    response = strtotal{i+8}(3:end-4);
    hit = str2num(strtotal{i-2}(1:end-1));
    counterbalance = str2num(strtotal{i+4}(1:end-1));
    rt = str2num(strtotal{i-4}(1:end-1));
    trial = str2num(strtotal{i+2}(1:end-1));
    
    ans = struct();
    ans.workerid = workerid;
    ans.assignmentid = assignmentid;
    ans.imageID = imageID;
    ans.response = response;
    ans.hit = hit+1;
    ans.counterbalance = counterbalance;
    ans.rt = rt;
    ans.trial = trial;

    if length(answer) > 0 
        if strcmp(answer(end).workerid,ans.workerid) && strcmp(answer(end).assignmentid,ans.assignmentid)
            answer = [answer ans];
        else
            subj.workerid = ans.workerid;
            subj.assignmentid = assignmentid;
            subj.numhits = length(answer);
            subj.answer = answer;
            subj.videorecord = 0;
            mturkData = [mturkData subj];
            answer = [];
            answer = [answer ans];
        end
    else
        answer = [answer ans];
    end
  
end
subj.workerid = ans.workerid;
subj.assignmentid = assignmentid;
subj.numhits = length(answer);
subj.answer = answer;
subj.videorecord = 0;
mturkData = [mturkData subj];

fclose(fid);
save(savefilename,'mturkData');

%% not done yet; extract dummy trials for quality check

fid = fopen(csvname);
out = textscan(fid,'%s');
strtotal = out{1,1};
pattern = 'http://kreiman.hms.harvard.edu/mturk/mengmi/dummy/';

mturkData = [];
answer = [];  
for i = 1:length(strtotal)

    str = strtotal{i,1};    
    k = strfind(str,pattern);    
    if isempty(k)
        continue;
    end
    
    if isempty(strfind(strtotal{i-12},'""uniqueid"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i-8},'{""current_trial"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i-5},'{""rt"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i-3},'""hit"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i-1},'""imageID"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i+1},'""trial"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i+3},'""counterbalance"":'))
        continue;
    end
    
    if isempty(strfind(strtotal{i+7},'""response"":'))
        continue;
    end
    
    strpart = strtotal{i-11};
    strpart = strsplit(strpart,':');
    workerid = strpart{1}(3:end);
    assignmentid = strpart{2}(1:end-3);
    imageID = strtotal{i}(3:end-3);
    response = strtotal{i+8}(3:end-4);
    hit = str2num(strtotal{i-2}(1:end-1));
    counterbalance = str2num(strtotal{i+4}(1:end-1));
    rt = str2num(strtotal{i-4}(1:end-1));
    trial = str2num(strtotal{i+2}(1:end-1));
    
    ans = struct();
    ans.workerid = workerid;
    ans.assignmentid = assignmentid;
    ans.imageID = imageID;
    ans.response = response;
    ans.hit = hit+1;
    ans.counterbalance = counterbalance;
    ans.rt = rt;
    ans.trial = trial;

    if length(answer) > 0 
        if strcmp(answer(end).workerid,ans.workerid) && strcmp(answer(end).assignmentid,ans.assignmentid)
            answer = [answer ans];
        else
            subj.workerid = ans.workerid;
            subj.assignmentid = assignmentid;
            subj.numhits = length(answer);
            subj.answer = answer;
            subj.videorecord = 0;
            mturkData = [mturkData subj];
            answer = [];
            answer = [answer ans];
        end
    else
        answer = [answer ans];
    end
  
end
subj.workerid = ans.workerid;
subj.assignmentid = assignmentid;
subj.numhits = length(answer);
subj.answer = answer;
subj.videorecord = 0;
mturkData = [mturkData subj];

DummymturkData = mturkData;
fclose(fid);

%% combine dummy trials to mturkdata
load(savefilename);
Dummyworkeridlist = extractfield(DummymturkData, 'workerid');
Dummyassignmentidlist = extractfield(DummymturkData, 'assignmentid');

for i = 1:length(mturkData)
    workerid = mturkData(i).workerid;
    assignmentid = mturkData(i).assignmentid;
    ind = find(strcmp(Dummyworkeridlist, workerid) & strcmp(Dummyassignmentidlist, assignmentid));
    if length(ind) < 1
        continue;
    end
    mturkData(i).dummy = DummymturkData(ind).answer;
end    

save(savefilename,'mturkData');









