function [ F, precision, recall ] = Fscore( predict, target, class )
%F_SCORE Summary of this function goes here
%   Detailed explanation goes here

a = double(predict==class);
b = double(target==class);

TP = sum((a.*b)==1);
FP = sum((a - b)==1);
FN = sum((b - a)==1);

precision = TP/(TP+FP);
recall = TP/(TP+FN);
F = 2*precision*recall/(precision + recall);
end

