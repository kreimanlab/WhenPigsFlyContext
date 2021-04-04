clear all; close all; clc;

PixToVA = 156/5; %pixels per visual angle in degrees
ObjSizeBin = [0.5 1 1.75 2.25 3.5 4.5 7 9]*PixToVA; 
ObjSizeBin = ObjSizeBin.^2; %threshold object size in pixels
NumBin = 4;

load(['Mat/VHhumanStats_noContext.mat']);

% for i = 1:length(VHhumanStats)
%     i
%     seg = imread(['/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/' VHhumanStats(i).segname]);
%     channel = seg(:,:,1);
%     numObjPixel = length(find(channel(:)>250));
%     
%     b = sqrt(numObjPixel)/PixToVA;   
%     VHhumanStats(i).Bin = b;
%     
%     infor = VHhumanStats(i);
%     area = (infor.right - infor.left) * ( infor.bottom - infor.top);
%     dva = sqrt(area)/PixToVA;
%     %infor.dva = dva;
%     VHhumanStats(i).dva = dva;
%         
% end
% save(['Mat/VHhumanStats_noContext.mat'],'VHhumanStats');

alldva = extractfield(VHhumanStats,'Bin');
hb = figure;

binrange = [0:1:10];
[N,edges] = histcounts(alldva,binrange);
plot(binrange(1:end-1), N./sum(N),'r-');
xlabel('Visual DVA');
ylabel('Proportion');

load(['Mat/mturk_expNoContext_results.mat']);
Binlist = extractfield(VHhumanStats, 'Bin');
overall = [];
counter = [];
for i = 1:10
    correctlist = extractfield(VHhumanStats(find(Binlist<i & Binlist>i-1)), 'GIFcorrect');
    overall = [overall nanmean(correctlist)];
    counter = [counter length(correctlist)];
end

hb1 = figure;
hold on;
plot([1:10], overall);
xlabel('dva');
ylabel('accuracy');
yyaxis right;
plot([1:10], counter);
ylabel('Num of responses')



