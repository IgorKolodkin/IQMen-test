X = table2array(train_data(:, [2:4]));
Y = train_data.VarName14;
%% create stepwiseglm
discrm = fitcdiscr(X, Y, 'DiscrimType', 'quadratic');
%%
%resuberror = resubLoss(discrm);
CM = confusionmat(discrm.Y,resubPredict(discrm));
P = predict(discrm, X);
[F1, precision, recall] = Fscore(P, Y, 0)
%%
md = fitcdiscr(X, Y, 'CrossVal','on', 'DiscrimType','pseudoQuadratic');
cvloss = kfoldLoss(md)
%% find model
DiscrimType = {'linear'
'quadratic'
'diagLinear'
'diagQuadratic'
'pseudoLinear'
'pseudoQuadratic'};
for i=1:length(DiscrimType)
md = fitcdiscr(X, Y, 'CrossVal','on','DiscrimType',DiscrimType{i});
cvDTloss(i) = kfoldLoss(md);
end
plot(cvDTloss)
% best is 'quadratic'
%% gamma fo linear
gamma = 1:20;
for i=1:length(gamma)
    md = fitcdiscr(X, Y, 'CrossVal','on','DiscrimType','quadratic');
    cvGloss(i) = kfoldLoss(md);
end
mean(cvGloss)
std(cvGloss)
%%
gscatter(X(:, 3),X(:, 4),Y,'krb','ov^',[],'off');

legend('0','1','Location','best')
hold on
%%
% Now, retrieve the coefficients for the quadratic boundary between the
% first and second classes (setosa and versicolor).
K = discrm.Coeffs(1,2).Const;
L = discrm.Coeffs(1,2).Linear;
Q = discrm.Coeffs(1,2).Quadratic;
 
% Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]'=0:
f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
    (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h3 = ezplot(f, [0, 1, 0, 1]); % Plot the relevant portion of the curve.
h3.Color = 'k';
h3.LineWidth = 2;
xlabel('Petal Length')
ylabel('Petal Width')
title('{\bf Quadratic Classification with Fisher Training Data}')
hold off