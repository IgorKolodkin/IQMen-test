% import
var10 = categorical(testS1.VarName10);
test_data = [test_data_num, table(var10)];
test_data = [test_data_num(:, 1:10), table(var10)];
test_data = [test_data_num(:, 1:9), table(var10)];
save('../Data/Processed_data/test_data.mat', 'test_data')
%%
load('../Data/Processed_data/test_data.mat')
load('../Data/Processed_data/train_data.mat')