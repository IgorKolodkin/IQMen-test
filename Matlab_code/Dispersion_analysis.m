% ������������� ������
%% �������� ����������� ������������� ��������� ���������� (respones variables)
data = table2array(train_data(:, 11:13));
alpha = 0.05;
NBins = round(sqrt(length(Y)));

test_var = data(:, 1);
[h,p,st] = chi2gof(test_var, 'Alpha', alpha, 'NBins', NBins)
test_result(1) = h;

test_var = data(:, 2);
[h,p,st] = chi2gof(test_var, 'Alpha', alpha, 'NBins', NBins);
test_result(2) = h;

test_var = data(:, 3);
[h,p,st] = chi2gof(test_var, 'Alpha', alpha, 'NBins', NBins);
test_result(3) = h;

disp(test_result)
% --------------- ������
% ������ ������ ��������� ���������� ����� ���������� ������������� �
% ������������ 95%

%% ������������� ������ ������ ����������� ���������� � 10-�� ���������.
X = train_data.VarName10;
Y = train_data.VarName13;
factors = categories(X);
%% �������� ��������� ���������
alpha = 0.05;
i=1;

for nf1 = 1:length(factors)
    for nf2 = nf1+1:length(factors)
        x1 = Y(X==factors(nf1));
        x2 = Y(X==factors(nf2));
        var_ratio = var(x1)/var(x2);
        
        pValue = min(2*fcdf(var_ratio, length(x1)-1, length(x2)-1), 2*(1 - fcdf(var_ratio, length(x1)-1, length(x2)-1)));
        test_result(i) = pValue > alpha;
        i = i+1;
    end
end
disp(test_result)
clear test_result
% --------------- ������
% ������� ��������� ��������� ��� ��������� �������� ����������� ��� ����
% ��� �������� � ������������ 95%.
% ����� ��������� ������������� ������
%% ANOVA1
volume = min(hist(X));
for i=1:length(factors)
    anova_data(1:volume, i) = datasample(Y(X==factors(i)), volume, 'Replace',false);
end
%%
p = anova1(anova_data)
%%
% �����: ��������� ������� (J �������(�������)) 
% ������� �� ������ �� �������������� �������(K-�������)p = 0.2817
%% save plot
plotname = 'Analysis of variance (ANOVA)';
fname = ['../Figures/Exploratory_figures/',plotname, '.png'];
saveas(gcf, fname, 'png')
fname = ['../Figures/Exploratory_figures/',plotname, '.fig'];
saveas(gcf, fname, 'fig')