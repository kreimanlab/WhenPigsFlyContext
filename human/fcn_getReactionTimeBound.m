function [meanRT, stdRT] = fcn_getReactionTimeBound(mturkData, display)
%FCN_GETREACTIONTIMEBOUND Summary of this function goes here
%   Detailed explanation goes here

RTlist = [];

for i = 1:length(mturkData)
    RT = extractfield(mturkData(i).answer, 'rt'); %in millisecs
    RTlist = [RTlist RT];
end

meanRT = mean(RTlist)/1000; %in secs
stdRT = std(RTlist)/1000; %in secs

if display == 1
    hb = figure;
    RTlist = RTlist/1000;
    binrange = [min(RTlist):5:100];
    [N,edges] = histcounts(RTlist,binrange);
    plot(binrange(1:end-1), N./sum(N),'r-');
    xlabel('Reaction Time (sec)');
    ylabel('Proportion');
end

end

