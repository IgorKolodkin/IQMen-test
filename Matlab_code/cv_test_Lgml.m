% prediction function given training/testing instances
fcn = @(Xtr, Ytr, Xte) predict(...
    GeneralizedLinearModel.fit(Xtr,Ytr,'linear','Distribution','Binomial'), Xte);

% perform cross-validation, and return average MSE across folds
mse = crossval('mse', R, Y, 'Predfun',fcn, 'kfold',10);

% compute root mean squared error
avrg_rmse = sqrt(mse)
%% misclassification cross-validation rate

mcrate = @(Xtr, Ytr, Xte, Yte) mcr_lr(...
    fitglm(Xtr,Ytr, 'Distribution', 'Binomial'), Xte, Yte);

% perform cross-validation, and return  MCRs across folds
mcr = crossval(mcrate, R, Y, 'kfold',10);

avrg_mcr = mean(mcr)
std_mcr = std(mcr)
%%
