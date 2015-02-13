%% load data
X = table2array(train_data(:, [1:4]));
% XJ =train_data.VarName10;
% factors = categories(XJ);
Y = train_data.VarName14;
%% plotmatrix
% subplot(1, 2, 1)
% plotmatrix(X(Y==1, :), '.r')
% subplot(1, 2, 2)
% plotmatrix(X(Y==0, :), '.b')
%% Cross-Validation-eval
model =
cp = cvpartition(Y,'k',10);
dtClassFun = @(xtrain,ytrain,xtest)(eval(classregtree(xtrain,ytrain),xtest));
dtCVErr  = crossval('mcr',X,Y,'predfun', dtClassFun,'partition',cp)

%% Cross-Validation-predict
cp = cvpartition(Y,'k',10);
dtClassFun = @(xtrain,ytrain,xtest)(predict(fitNaiveBayes(xtrain,ytrain),xtest));
dtCVErr  = crossval('mcr',X,Y,'predfun', dtClassFun,'partition',cp)

%% ========================= NaiveBayes
NBModel = fitNaiveBayes(X,Y);
sum(predict(NBModel, X) ~= Y)/length(Y)
%% bosting 
Ensemble = fitensemble(X,Y,'Subspace',NLearn,'Discriminant')
%% cv boosting
NLearn = 50;
Ensemble = @(X, Y)(fitensemble(X,Y,'RUSBoost',NLearn,'Discriminant'));
cp = cvpartition(Y,'k',10);
dtClassFun = @(xtrain,ytrain,xtest)(predict(Ensemble(xtrain,ytrain),xtest));
dtCVErr  = crossval('mcr',X,Y,'predfun', dtClassFun,'partition',cp)

%% ========================= knn

%%
for nn = 1:20
    mdl = fitcknn(X,Y,'NumNeighbors',nn);
    rloss(nn) = resubLoss(mdl);
    cp = cvpartition(Y,'k',10);
    dtClassFun = @(xtrain,ytrain,xtest)(predict(fitcknn(xtrain,ytrain,'NumNeighbors',2),xtest));
    dtCVErr(nn)  = crossval('mcr',X,Y,'predfun', dtClassFun,'partition',cp);
end
plot(dtCVErr)
%% knn expml

[N,D] = size(X);
resp = unique(Y);
rng(8000,'twister') % for reproducibility
K = round(logspace(0,log10(N),10)); % number of neighbors
cvloss = zeros(numel(K),1);
for k=1:numel(K)
    knn = fitcknn(X,Y,...
        'NumNeighbors',K(k),'CrossVal','On');
    cvloss(k) = kfoldLoss(knn);
end
figure; % Plot the accuracy versus k
semilogx(K,cvloss);
xlabel('Number of nearest neighbors');
ylabel('10 fold classification error');
title('k-NN classification');
%% esemple
X = table2array(train_data(:, [1:9]));
NPredToSample = round(linspace(1,D,5)); % linear spacing of dimensions
cvloss = zeros(numel(NPredToSample),1);
learner = templateKNN('NumNeighbors',7);
for npred=1:numel(NPredToSample)
    subspace = fitensemble(X,Y,'Subspace',200,learner,...
        'NPredToSample',NPredToSample(npred),'CrossVal','On');
    cvloss(npred) = kfoldLoss(subspace);
    fprintf('Random Subspace %i done.\n',npred);
end
figure; % plot the accuracy versus dimension
plot(NPredToSample,cvloss);
xlabel('Number of predictors selected at random');
ylabel('10 fold classification error');
title('k-NN classification with Random Subspace');
%% esn tree
[N,D] = size(X);
resp = unique(Y);
rng(8000,'twister') % for reproducibility

K = round(logspace(1,log10(N/10),10)); % number of neighbors
cvloss = zeros(numel(K),1);
for k=1:numel(K)
    templ = templateTree('MinLeaf',round(size(X,1)/50));
    ens = fitensemble(X,Y,'AdaBoostM1',K(k),templ,'CrossVal','On');
    cvloss(k) = kfoldLoss(ens);
end
figure; % Plot the accuracy versus k
semilogx(K,cvloss);
xlabel('Number of trees');
ylabel('10 fold classification error');
title('tree classification');
%% glm
mdl = fitglm(X,(Y-0.5)*2,'quadratic')
sum(double(predict(mdl, X)>0) ~= Y)/length(Y)