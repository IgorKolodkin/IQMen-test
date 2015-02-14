function [ mcr ] = mcr_lr( md, Xte, Yte )
%MCR_LR Summary of this function goes here
%   Detailed explanation goes here

out = predict_lr(md, Xte);

mcr = sum(out~= Yte)/length(Yte);

end

