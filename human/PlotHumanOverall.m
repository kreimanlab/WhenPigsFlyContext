clear all; close all; clc;

load(['Mat/VHhumanStats_GTresult.mat']);
classidlist = extractfield(VHhumanStats, 'classid');
classuniqueidlist = unique(classidlist);
all = {};

startcolor = [0.8706    0.9216    0.9804];
overcolor = [0.3922    0.4745    0.6353];
NumConds = 6-2;
barcolor = [linspace(startcolor(1),overcolor(1),NumConds)', linspace(startcolor(2),overcolor(2),NumConds)', linspace(startcolor(3),overcolor(3),NumConds)'];
bboxcolor = [1 1 1];
fccolor = [0 0 0];

barcolor = [bboxcolor; barcolor; fccolor];

% barcolor = [1     1     1;...
%     0.8314    0.8157    0.7843;...    
%     0.5020    0.5020    0.5020;...
%     0.3314    0.3157    0.3843;...
%     0     0     0];
expnamelist = {'expNormal', 'expGravity', 'expAnomaly', 'expNoContext', 'expSize'}; %,'expMaterial'
condnamelist = {'NormalContext', 'Gravity(G)', 'Co-occurrence(C)', 'G+C', 'NoContext', 'Size'};

mturk_mean = []; mturk_std = [];
counter = 1;
for exp = 1:length(expnamelist)
    
    expname = expnamelist{exp}
    load(['Mat/mturk_' expname '_results.mat']);
    
    if strcmp(expname,'expAnomaly')
        
        %co-occurence condition
        dvalist = extractfield(VHhumanStats, 'dva');
        condwalllist = extractfield(VHhumanStats, 'condWall');
        overall = extractfield(VHhumanStats(find(condwalllist == 0 & dvalist <= 2)), 'GIFcorrect');   

        mturk_mean = [mturk_mean nanmean(overall)];
        mturk_std = [mturk_std nanstd(overall)/sqrt(length(overall))];

        dvalist = extractfield(VHhumanStats, 'dva');
        condwalllist = extractfield(VHhumanStats, 'condWall');
        overall = extractfield(VHhumanStats(find(condwalllist == 0 & dvalist > 2)), 'GIFcorrect');   

        mturk_mean = [mturk_mean nanmean(overall)];
        mturk_std = [mturk_std nanstd(overall)/sqrt(length(overall))];
        
        %G+C condition
        dvalist = extractfield(VHhumanStats, 'dva');
        condwalllist = extractfield(VHhumanStats, 'condWall');
        overall = extractfield(VHhumanStats(find(condwalllist == 1 & dvalist <= 2)), 'GIFcorrect');   

        mturk_mean = [mturk_mean nanmean(overall)];
        mturk_std = [mturk_std nanstd(overall)/sqrt(length(overall))];

        dvalist = extractfield(VHhumanStats, 'dva');
        condwalllist = extractfield(VHhumanStats, 'condWall');
        overall = extractfield(VHhumanStats(find(condwalllist == 1 & dvalist > 2)), 'GIFcorrect');   

        mturk_mean = [mturk_mean nanmean(overall)];
        mturk_std = [mturk_std nanstd(overall)/sqrt(length(overall))];
        
    else
    
        dvalist = extractfield(VHhumanStats, 'dva');
        overall = extractfield(VHhumanStats(find(dvalist <= 2)), 'GIFcorrect');   

        mturk_mean = [mturk_mean nanmean(overall)];
        mturk_std = [mturk_std nanstd(overall)/sqrt(length(overall))];

        dvalist = extractfield(VHhumanStats, 'dva');
        overall = extractfield(VHhumanStats(find(dvalist > 2)), 'GIFcorrect');   

        mturk_mean = [mturk_mean nanmean(overall)];
        mturk_std = [mturk_std nanstd(overall)/sqrt(length(overall))];
    end
        
end

mturk_mean = reshape(mturk_mean, [2 length(expnamelist)+1]);
mturk_std = reshape(mturk_std, [2 length(expnamelist)+1]);

mturk_mean = mturk_mean(:,[5 2 3 4 6 1]);
mturk_std = mturk_std(:,[5 2 3 4 6 1]);

hb = figure('units','pixels');
hold on;
% mturk_mean = subjplot_mean; %all elements in a row belong to the same group; size(A, 1) is the number of groups
% mturk_std = subjplot_std;

ngroups = size(mturk_mean, 1);
nbars = size(mturk_mean, 2);
H = bar(mturk_mean);
for b = 1:nbars
    set(H(b),'FaceColor',barcolor(b,:));
    %set(H(b),'EdgeColor',edgecolor(b,:));
    set(H(b),'LineWidth',2);
end

% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, mturk_mean(:,i), mturk_std(:,i), 'k.');
end
%errorbar(xaxis,ones(1,length(xaxis))*1/80,zeros(1,length(xaxis)),'k--','LineWidth',2);
xlim([0.5 ngroups+0.5]);
ylim([0 1]);
hold off
%legend(LegName,'Location','Northwest','FontSize', 12);

LegName = condnamelist([5 2 3 4 6 1]); 
legend(LegName,'Location','Northeastoutside','FontSize', 12);
XtickLabelName = {'Target<=2 dva','Target>2 dva'};

%xlabel('Context Object Ratio','FontSize',12);
xticks([1:ngroups]);
set(gca,'YTick',[0:0.2:1]);
set(gca, 'TickDir', 'out')
%set(gca,'XTickLength',[0 0]);
%set(gca,'XTick',[]);
set(gca,'XTickLabel',char(XtickLabelName),'FontSize',15);
ylabel('Top-1 Accuracy','FontSize', 15);
title('mturk; humans (36 classes)','FontSize', 15);
legend('boxoff'); 

%get(hb,'Position')
set(hb,'Position',[1035         572         808         408]);
% printpostfix = '.eps';
% printmode = '-depsc'; %-depsc
printpostfix = '.png';
printmode = '-dpng'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figures/fig_human_mturk_overall' printpostfix],printmode,printoption);


