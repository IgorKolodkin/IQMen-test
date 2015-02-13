X = table2array(train_data(:, [1, 2]));
XJ =train_data.VarName10;
Y = train_data.VarName13;
factors = categories(XJ);
%%
% f=3;
% x = X(XJ==factors{fq(f)}, :);
% y = Y(XJ==factors{fq(f)}, :);
%% stepwise regress
% fq = [1,2,4:7];
X = table2array(train_data(:, 1:9));
XJ =train_data.VarName10;
Y = train_data.VarName13;
factors = categories(XJ);
fq = [1:7];
rmm =cell(1, length(fq));
for f=1:length(fq)
    x = X(XJ==factors{fq(f)}, :);
    y = Y(XJ==factors{fq(f)}, :);
    rmm{f} = LinearModel.stepwise(x, y, 'constant', 'Upper','poly333333333');
    rmm{f}
    STAT(f, :) = [lm{f}.RMSE, lm{f}.Rsquared.Adjusted];
    v(f) = length(x);
    param(f, :) = [mean(y), std(y)];
end
%%
for f=1:length(fq)
    dep{f,1} = rmm{f}.Formula.LinearPredictor;
end
% save('./reg_model_m.mat', 'rmm')
%% просто regress
fq = [1:7];
for f=1:length(fq)
    x = X(XJ==factors{fq(f)}, :);
    R = [ones(length(x), 1), x, x(:, 1).*x(:, 2), x.^2];
    y = Y(XJ==factors{fq(f)}, :);
    [b(:, f),bint{f},r{f},rint{f},stats{f}] = regress(y,R);
    
end
%% fitlm
fq = [1:7];
for f=1:length(fq)
    x = X(XJ==factors{fq(f)}, :);
    R = [x, x(:, 1).*x(:, 2), x.^2];
    y = Y(XJ==factors{fq(f)}, :);
    lm{f} = fitlm(R,y,'linear');
    lm{f};
    STAT(f, :) = [lm{f}.RMSE, lm{f}.Rsquared.Adjusted];
    v(f) = length(x);
    param(f, :) = [mean(y), std(y)]
end
%% cv
for f=1:length(fq)
    x = X(XJ==factors{fq(f)}, [1,2]);
    R = [x, x(:, 1).*x(:, 2), x.^2];
    y = Y(XJ==factors{fq(f)}, :);
    
    cp = cvpartition(y,'k',10);
    dtClassFun = @(xtrain,ytrain,xtest)(predict(fitlm(xtrain,ytrain),xtest));
    dtCVErr(f)  = crossval('mcr',R,y,'predfun', dtClassFun,'partition',cp);
end
%% plot regress
f=1;
x1 = X(XJ==factors{fq(f)}, 1)';
x2 = X(XJ==factors{fq(f)}, 2)';
y = Y(XJ==factors{fq(f)}, :)';
c = b(:, f);
scatter3(x1,x2,y,'filled')
hold on
x1fit = min(x1):0.05:max(x1);
x2fit = min(x2):0.05:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = c(1) + c(2)*X1FIT + c(3)*X2FIT + c(4)*X1FIT.*X2FIT + c(5)*X1FIT.^2 + c(6)*X2FIT.^2;
mesh(X1FIT,X2FIT,YFIT)
view(50,10)