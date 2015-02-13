% Small multiples
plotname = 'Scatter plot matrix';
X = table2array(train_data(:, 1:9));
Y = table2array(train_data(:, 13));
figure
% pos = get(gcf, 'Position');
% set(gcf, 'Position', [pos(1)-500 pos(2) 1200, 300]); %<- Set size
plotmatrix(X,Y)
title(plotname)
xlabel('Predictor Variable')
ylabel('Response Variable')
%%
fname = ['../Figures/Exploratory_figures/',plotname, '.png'];
saveas(gcf, fname, 'png')
fname = ['../Figures/Exploratory_figures/',plotname, '.fig'];
saveas(gcf, fname, 'fig')
%%
plotname = 'Regression fitted Residuals J4';
fname = ['../Figures/Final_figures/',plotname, '.png'];
saveas(gcf, fname, 'png')
%% Гистограммы всех признаков
 X = table2array(test_data(:, 1:9));
%X = table2array(train_data(:, 1:9));
figure
for i =1:9
   subplot(3, 3, i)
   hist(X(1:end, i))
end

%% Гистограммы признаков в зависимости от L
%X = table2array(test_data(:, 1:9));
X = table2array(train_data(train_data.VarName14==0, 1:9));
figure
for i =1:9
   subplot(3, 3, i)
   hist(X(1:end, i))
end

X = table2array(train_data(train_data.VarName14==1, 1:9));
figure
for i =1:9
   subplot(3, 3, i)
   hist(X(1:end, i))
end
%% J from L{0, 1}
Y = train_data.VarName14;
X = train_data.VarName10;
[a(:, 1), b(:, 1)] = hist(X(Y==1));
[a(:, 2), b(:, 2)] = hist(X(Y==0));
bar(a)
%%
