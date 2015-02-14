load('../Data/Processed_data/test_data.mat');
%%
Xtest = table2array(test_data(:, 1:9));
DM = mdl.Formula.Terms(:, 1:end-1);
A = x2fx(Xtest, DM);
Rtest = A(:, 2:end);
%%
Ntest = predict_lr(mdl, Rtest);


%%
[ Mtest ] = predict_M(lm,  test_data )
%%
[ P ] = predict_M(lm,  train_data );
mean(abs(P-Y))
