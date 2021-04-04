clear all; close all; clc;

load(['Mat/VHhumanStats_GTresult.mat']);
classidlist = extractfield(VHhumanStats, 'classid');
classuniqueidlist = unique(classidlist);
all = {};

barcolor = [1     1     1;...
    0.8314    0.8157    0.7843;...
    0.5020    0.5020    0.5020;...
    0     0     0];
expnamelist = {'expNormal', 'expGravity', 'expAnomaly', 'expNoContext', 'expSize'}; %,'expMaterial'
    
mturk_mean = []; mturk_std = [];
counter = 1;
for exp = 1:length(expnamelist)
    expname = expnamelist{exp};
    load(['Mat/mturk_' expname '_results.mat']);
    
    mturk_mean = [mturk_mean nanmean(overall)];
    mturk_std = [mturk_std nanstd(overall)/sqrt(length(overall))];
    all{counter} = overall;
    counter = counter+1;
    
    if exp == 3
        mturk_mean = [mturk_mean nanmean(overall_YW)];
        mturk_std = [mturk_std nanstd(overall_YW)/sqrt(length(overall_YW))];
        all{counter} = overall_YW;
        counter = counter+1;
        
        mturk_mean = [mturk_mean nanmean(overall_NW)];
        mturk_std = [mturk_std nanstd(overall_NW)/sqrt(length(overall_NW))];
        all{counter} = overall_NW;
        counter = counter+1;
    end
    
    if exp == 4
        mturk_mean = [mturk_mean nanmean(overall_condG)];
        mturk_std = [mturk_std nanstd(overall_condG)/sqrt(length(overall_condG))];
        all{counter} = overall_condG;
        counter = counter+1;
        
        mturk_mean = [mturk_mean nanmean(overall_condSP)];
        mturk_std = [mturk_std nanstd(overall_condSP)/sqrt(length(overall_condSP))];
        all{counter} = overall_condSP;
        counter = counter+1;
    end
    
    if exp == 5
        mturk_mean = [mturk_mean nanmean(overall_S2)];
        mturk_std = [mturk_std nanstd(overall_S2)/sqrt(length(overall_S2))];
        all{counter} = overall_S2;
        counter = counter+1;
        
        mturk_mean = [mturk_mean nanmean(overall_S3)];
        mturk_std = [mturk_std nanstd(overall_S3)/sqrt(length(overall_S3))];
        all{counter} = overall_S3;
        counter = counter+1;
        
        mturk_mean = [mturk_mean nanmean(overall_S4)];
        mturk_std = [mturk_std nanstd(overall_S4)/sqrt(length(overall_S4))];
        all{counter} = overall_S4;
        counter = counter+1;
    end
        
end

hb = figure; hold on;
ngroups = size(mturk_mean, 1);
nbars = size(mturk_mean, 2);
NumTypes = counter-1;
NumVisualBin = 1;
xaxis = [1:NumTypes]; 

xticklabelstring =str2mat('Normal','Gravity','Anomaly', 'AnomalyYW','AnomalyNW',...
    'NoContext','NoContextG','NoContextSP', 'Size','Size(2)','Size(3)','Size(4)'); %,'Material'
%legendstring = {'[0.5 1]','[1.75 2.25]','[3.5 4.5]','[7 9]'};
H = bar(mturk_mean);
for b = 1:NumVisualBin
    set(H(b),'FaceColor',barcolor(b,:));
end

% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = i; %(1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, mturk_mean(:,i), mturk_std(:,i), 'k.','HandleVisibility','off');
end
plot([0.5 1:nbars nbars+0.5], mturk_mean(1)*ones(size([0.5 1:nbars nbars+0.5])), 'k--','LineWidth',2,'HandleVisibility','off');
%errorbar(xaxis,ones(1,length(xaxis))*1/80,zeros(1,length(xaxis)),'k--','LineWidth',2,'HandleVisibility','off');
xlim([0.5 NumTypes+0.5]);
ylim([0 1.0]);
hold off
%legend(legendstring,'Location','Northeast','FontSize', 12);

%xlabel(xlabelstring,'FontSize',12);
set(gca,'YTick',[0:0.2:1]);
set(gca,'XTick',(xaxis));
set(gca,'TickLength',[0 0]);
set(gca,'XTickLabel',xticklabelstring);
ylabel('Top-1 Accuracy','FontSize', 12);
%title( [expname ' (mturk overall); number of trials on average: ' num2str(NUMTRIALS)],'FontSize', 12);

%compute stats and put asterisks
posy = 0.9
for i = 2:nbars
    [h pval] = ttest2(all{1}, all{i});
    if pval < 0.05
        textstring = '*';
    else
        textstring = 'ns';
    end
    text(i, posy,textstring,'FontSize',14,'FontWeight','Bold');
end

%get(hb,'Position')
set(hb,'Position',[1035         572         808         408]);
printpostfix = '.eps';
printmode = '-depsc'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figures/fig_human_mturk_overall' printpostfix],printmode,printoption);


