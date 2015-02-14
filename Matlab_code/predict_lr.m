function [ out ] = predict_lr( mdl, Xte)
%PREDICT_LR Summary of this function goes here
%   Detailed explanation goes here
b = mdl.Coefficients.Estimate;

% T = mdl.Formula.Terms;
% Rte = x2fx(Xte, T(:, 1:end-1));

P = glmval(b, Xte, 'logit');

out = double(P>0.5);
end

