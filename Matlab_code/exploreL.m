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
%%
% Small multiples
Y = table2array(train_data(:, 14));
X = table2array(train_data(:, 2:4));
% X(:, 2) = X(:, 2) - 0.25;

X(:, 1) = X(:, 1) - mean(X(:, 1));
X(:, 2) = X(:, 2) - mean(X(:, 2));
X(:, 3) = X(:, 3) - mean(X(:, 3));

[X(:, 1), X(:, 2)] = cart2pol(X(:, 1),X(:, 2));

%
p = Y==0;
figure
subplot(1, 2,1)
plotmatrix(X(p, :), '.r')
title('Class 0')
subplot(1, 2, 2)
plotmatrix(X(~p, :),'.b')
title('Class 1')
%%