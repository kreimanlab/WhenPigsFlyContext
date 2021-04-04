clear all; close all; clc;

load(['Mat/resultMat_VHcoco.mat']);

mturk_mean = result_mean;
mturk_std = result_std;

startcolor = [0.8706    0.9216    0.9804];
overcolor = [0.3922    0.4745    0.6353];
%NumConds = 2-2;
%barcolor = [linspace(startcolor(1),overcolor(1),NumConds)', linspace(startcolor(2),overcolor(2),NumConds)', linspace(startcolor(3),overcolor(3),NumConds)'];
bboxcolor = [1 1 1];
fccolor = [0 0 0];

%barcolor = [bboxcolor; barcolor; fccolor];
barcolor = [bboxcolor; fccolor];


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

LegName = {'Out-of-Context','Normal-Context'}; 
legend(LegName,'Location','Northeast','FontSize', 12);
XtickLabelName = {'CRTNet','CATNet','FastRCNN','DenseNet'};
xtickangle(10);
%xlabel('Context Object Ratio','FontSize',12);
xticks([1:ngroups]);
set(gca,'YTick',[0:0.2:1]);
set(gca, 'TickDir', 'out')
%set(gca,'XTickLength',[0 0]);
%set(gca,'XTick',[]);
set(gca,'XTickLabel',char(XtickLabelName),'FontSize',15);
ylabel('Top-1 Accuracy','FontSize', 15);
%title('expA What (mturk)','FontSize', 12);
legend('boxoff'); 

%get(hb,'Position')
set(hb,'Position',[680   743   538   346]);
% printpostfix = '.eps';
% printmode = '-depsc'; %-depsc
printpostfix = '.pdf';
printmode = '-dpdf'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figures/fig_unrel_overall' printpostfix],printmode,printoption);
