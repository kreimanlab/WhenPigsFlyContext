clear all; clc; close all;

rowleg = {'Size1','Size2','Size3','Size4'};
colleg = {'fullContext','congruent','incongruent','minimalContext'};

overall_mean = [];
overall_std = [];

matpath = ['../Cut-Paste/cvpr_results/'];

%% humans
expname = 'expH';
modelname = 'mturk';
load([matpath 'Mat/stats_' expname '_' modelname '.mat']);
subjplot_mean = subjplot_mean(:,[3 1:2 4])';
subjplot_std = subjplot_std(:,[3 1:2 4])';
overall_mean = [overall_mean subjplot_mean(:)];
overall_std = [overall_std subjplot_std(:)];

%% ours
selected = 3;
baseline = {'fastrcnn','densenet','ours'};
path = ['../Cut-Paste/' baseline{selected} '_results/'];
load([path 'resultMat.mat']);
subjplot_mean = reshape(mean, [4 4])';
subjplot_std = reshape(sem, [4 4])';
overall_mean = [overall_mean subjplot_mean(:)];
overall_std = [overall_std subjplot_std(:)];


%% CATNet
expname = 'expH';
modelname = 'clicknet_noalphaloss';
load([matpath 'Mat/stats_' expname '_' modelname '.mat']);
subjplot_mean = subjplot_mean(:,[3 1:2 4])';
subjplot_std = subjplot_std(:,[3 1:2 4])';
overall_mean = [overall_mean subjplot_mean(:)];
overall_std = [overall_std subjplot_std(:)];

%% fastrcnn
selected = 1;
baseline = {'fastrcnn','densenet'};
path = ['../Cut-Paste/' baseline{selected} '_results/'];
load([path 'resultMat.mat']);
subjplot_mean = reshape(mean, [4 4])';
subjplot_std = reshape(sem, [4 4])';
overall_mean = [overall_mean subjplot_mean(:)];
overall_std = [overall_std subjplot_std(:)];

%% densenet
selected = 2;
baseline = {'fastrcnn','densenet'};
path = ['../Cut-Paste/' baseline{selected} '_results/results/'];
load([path 'resultMat.mat']);
subjplot_mean = reshape(mean, [4 4])';
subjplot_std = reshape(sem, [4 4])';
overall_mean = [overall_mean subjplot_mean(:)];
overall_std = [overall_std subjplot_std(:)];

%% to latex
matrix = overall_mean'*100;
filename = '/home/mengmi/Desktop/temp.tex';
matrix2latex(matrix, filename, 'format', '%-3.1f');
filename = '/home/mengmi/Desktop/temp_std.tex';
matrix = overall_std'*100;
matrix2latex(matrix, filename, 'format', '(%-3.1f)');



% %% plot
% hb = figure;
% hold on;
% colorlist = jet(size(overall_mean,2)-1);
% for i = 1:(size(overall_mean,2)-1)
%     %plot( overall_mean(:,i+1), overall_mean(:,1),'o','color',colorlist(i,:));
%     
%     %plot linear line
%     coefficients = polyfit(overall_mean(:,i+1), overall_mean(:,1), 1);
%     xFit = linspace(min(overall_mean(:,i+1)), 1, 1000);
%     % Get the estimated yFit value for each of those 1000 new x locations.
%     yFit = polyval(coefficients , xFit);
%     plot(xFit, yFit, 'Color',colorlist(i,:),'LineWidth',2,'LineStyle','--'); % Plot fitted line.
% end
% plot([0 1],[0 1],'k-','LineWidth',2);
% 
% for i = 1:(size(overall_mean,2)-1)
%     plot( overall_mean(:,i+1), overall_mean(:,1),'o','color',colorlist(i,:));
%     
% end
% 
% legend({'CATNet','FastRCNN','DenseNet','Identity'},'FontSize',12,'FontWeight','bold','Location','southeastoutside');
% 
% 
% yticks([0:0.2:1]);
% xticks([0:0.2:1]);
% xlim([0 1]);
% ylim([0 1]);
% ylabel('Human (Accuracy)','FontSize',12,'FontWeight','bold');
% xlabel('Model (Accuracy)','FontSize',12,'FontWeight','bold');







