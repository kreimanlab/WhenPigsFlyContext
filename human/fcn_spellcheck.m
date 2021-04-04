function [ check ] = fcn_spellcheck(guessword, wordlist)
%FCN_SPELLCHECK Summary of this function goes here
%   Detailed explanation goes here

thres = cellfun(@length,wordlist);
thres = floor(thres*0.8);

boolstore = zeros(1,length(wordlist));
for w = 1:length(guessword)
    character = guessword(w);
    TF = strfind(wordlist,character);
    wordlist = strrep(wordlist,character,'');
    TF = cellfun(@isempty,TF);
    TF = ~TF;
    boolstore = boolstore + TF;
end
boolstorecheck = (boolstore>thres);

if sum(boolstorecheck)>0
    check  = 1;
else
    check = 0;
end

