function [ mdl, R ] = fit_logit_reg( X, Y )
%LOGIT_RATE Summary of this function goes here
%   Detailed explanation goes here

Tinit = [0 0 0, 0 0 0, 0 0 0 0; ...
    0 0 1, 0 0 0, 0 0 0 0; ...
    0 0 0, 1 0 0, 0 0 0 0 ];

mdl = stepwiseglm(X,Y, Tinit,'upper','quadratic', 'Distribution', 'Binomial');

DM = mdl.Formula.Terms(:, 1:end-1);
A = x2fx(X, DM);
R = A(:, 2:end);
end

