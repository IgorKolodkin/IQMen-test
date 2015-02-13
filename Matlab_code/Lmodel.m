X = table2array(train_data(:, [1:4]));
% XJ =train_data.VarName10;
% factors = categories(XJ);
Y = train_data.VarName14;

% subplot(1, 2, 1)
% plotmatrix(X(Y==1, :), '.r')
% subplot(1, 2, 2)
% plotmatrix(X(Y==0, :), '.b')
%%
obj = fitcdiscr(X,Y,'DiscrimType','quadratic');
resubLoss(obj)
%% tree
t = classregtree (X, Y, 'CategoricalPredictors', 3);
%%
cp = cvpartition(Y,'k',10);
dtClassFun = @(xtrain,ytrain,xtest)(eval(classregtree(xtrain,ytrain),xtest));
dtCVErr  = crossval('mcr',X,Y, ...
          'predfun', dtClassFun,'partition',cp)
      %%
resubcost = test(t,'resub');
[cost,secost,ntermnodes,bestlevel] = test(t,'cross',X,Y);
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
figure(gcf);
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
legend('Cross-validation','Resubstitution')
%%
[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 20], [cutoff cutoff], 'k:')
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
hold off
%%
pt = prune(t,bestlevel);
view(pt)
%% NaiveBayes
X = table2array(train_data(:, [1:4]));
Y = train_data.VarName14;
% train
NBModel = fitNaiveBayes(X,Y);
% cv
cp = cvpartition(Y,'k',10);
dtClassFun = @(xtrain,ytrain,xtest)(predict(fitNaiveBayes(xtrain,ytrain),xtest));
dtCVErr  = crossval('mcr',X,Y,'predfun', dtClassFun,'partition',cp)
%% bosting
X = table2array(train_data(:, [1:4]));
Y = train_data.VarName14;

@Ensemble = fitensemble(X,Y,'Subspace',NLearn,'Discriminant')
%%
NLearn = 50;
Ensemble = @(X, Y)(fitensemble(X,Y,'RUSBoost',NLearn,'Discriminant'));
cp = cvpartition(Y,'k',10);
dtClassFun = @(xtrain,ytrain,xtest)(predict(Ensemble(xtrain,ytrain),xtest));
dtCVErr  = crossval('mcr',X,Y,'predfun', dtClassFun,'partition',cp)