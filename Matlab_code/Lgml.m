X = table2array(train_data(:, [1:9]));
Y = train_data.VarName14;
%% glm
mdl = fitglm(X,Y,'linear', 'Distribution', 'poisson','link','log');
sum(double(predict(mdl, X)>0.5) ~= Y)/length(Y)
%% ROC AUC F1
P = predict(mdl, X);
[Xroc,Yroc,~,AUC] = perfcurve(Y,P,1);
plot(Xroc, Yroc)
[F1, precision, recall] = Fscore(double(P>0.44), Y, 1);
%%
i=1;
for t = 0:0.01:1
    [F1(i), precision, recall] = Fscore(double(P>t), Y, 1);
    i=i+1;
end
plot(0:0.01:1, F1, '.-')
%% stepwiseglm
T = [0 0 0, 0 0 0, 0 0 0 0; ...
    0 0 1, 0 0 0, 0 0 0 0; ...
    0 0 0, 1 0 0, 0 0 0 0 ];
mdl = stepwiseglm(X,Y, T,'upper','quadratic', 'Distribution', 'poisson','link','log');
sum(double(predict(mdl, X)>0.5) ~= Y)/length(Y)
% ROC AUC F1
P = predict(mdl, X);
[Xroc,Yroc,~,AUC] = perfcurve(Y,P,1);
%% eval model
plotSlice(mdl)
plotDiagnostics(mdl)
