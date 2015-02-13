X = table2array(train_data(:, [1:4]));
% XJ =train_data.VarName10;
% factors = categories(XJ);
Y = train_data.VarName14;
%%
p = Y==0;
f1 = 3;f2 = 4;
hold on;
plot(X(p, f1), X(p, f2), '.r')
plot(X(~p, f1), X(~p, f2), '.g')