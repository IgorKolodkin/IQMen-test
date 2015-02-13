
%%
X = table2array(train_data(:, [2:3]));
% XJ =train_data.VarName10;
% factors = categories(XJ);
Y = train_data.VarName14;
X(:, 1) = X(:, 1) - 0.5;
X(:, 2) = X(:, 2) - 0.25;
[THETA,RHO] = cart2pol(X(:, 1),X(:, 2));
figure
subplot(1, 2, 1)
polar(THETA(Y==0),RHO(Y==0), '.r')
subplot(1, 2, 2)
polar(THETA(Y==1),RHO(Y==1), '.g')
%
figure
subplot(1, 2, 1)
histfit(THETA(Y==0))
subplot(1, 2, 2)
histfit(THETA(Y==1))
%
figure
subplot(1, 2, 1)
histfit(RHO(Y==0))
subplot(1, 2, 2)
histfit(RHO(Y==1))
%%
scatter3(X(p, 1), X(p, 2), X(p, 3), '.r')
hold on
scatter3(X(~p, 1), X(~p, 2), X(~p, 3), '.g')