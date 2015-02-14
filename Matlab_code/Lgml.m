X = table2array(train_data(:, [1:9]));
Y = train_data.VarName14;
%% create stepwiseglm
[lrm, R] = fit_logit_reg(X, Y);
%%
P = predict_lr(lrm, R);
%%
mcr = mcr_lr(lrm, R, Y)
%% ;
[Xroc,Yroc,~,AUC] = perfcurve(Y,P,1);
plot(Xroc, Yroc)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC for classification for glm')
%%
i=1;
thr = 0:0.01:1;
for t = 1:length(thr)
    [F1(t), precision, recall] = Fscore(double(P>thr(t)), Y, 1);
end
%%
[a, b] = max(F1);
THR = thr(b);
%%
[F1, precision, recall] = Fscore(P, Y, 0)
%%
plot(0:0.01:1, F1, '.-')
xlabel('Threshold')
ylabel('F1 score')
title('F1 score')

%% eval model
plotSlice(mdl)
plotDiagnostics(mdl)
%%
f=1;
x1 = X(:, 3)';
x2 = X(:, 4)';
y = Y';
c = lrm.Coefficients.Estimate;
scatter(x1,x2,'filled')
hold on
x1fit = min(x1):0.05:max(x1);
x2fit = min(x2):0.05:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = R*c;
mesh(X1FIT,X2FIT,YFIT)
view(50,10)