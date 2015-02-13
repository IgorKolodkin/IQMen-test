%% Зависимость всех and M
X = table2array(train_data(:, 1:9));
J = train_data.VarName10;
factors = categories(J);
Y = train_data.VarName13;
%%
colors = {'r', 'g', 'b', 'black', 'y', 'p', 'm'};
figure
for sp = 1:size(X, 2)
    subplot(1, size(X, 2), sp)
    for f = 1:length(factors)
        bf = J == factors(f);
        scatter(X(bf, sp), Y(bf), colors{f})
        hold on
    end
    hold off
end

%% Зависимость всех and L
X = table2array(train_data(:, 1:9));
J = train_data.VarName10;
factors = [0, 1];
Y = train_data.VarName14;

colors = {'r', 'b'};
figure
for sp = 1:size(X, 2)
    subplot(3, 3, sp)
    for f = 1:length(factors)
        bf = Y == factors(f);
        [a(:, f), b(:, f)] = hist(X(bf, sp));
    end
    bar(b, a)
end
%% ANOVA for L
factors = [0, 1];
F = train_data.VarName14;
volume = min(sum(F), length(F) - sum(F));
for j=1:9
    X = table2array(train_data(:, j));
    anova_data = zeros(volume, length(factors));
    for i=1:length(factors)
        anova_data(1:volume, i) = datasample(X(F==factors(i)), volume, 'Replace',false);
    end
    p(j) = anova1(anova_data);
end
%%

