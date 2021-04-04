function [correct] = fcn_checkDummyResponse(dummy, GT)
%FCN_CHECKDUMMYRESPONSE Summary of this function goes here
%   Detailed explanation goes here

    correct = 0;
    for i = 1:length(dummy)
        res = dummy(i).response;
        x = dummy(i).imageID;
        x = str2num(x(end-4));
        if length(strcmp(GT{x},res)) > 0
            correct = correct + 1;
        end
    end


end

