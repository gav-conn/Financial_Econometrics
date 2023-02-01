%% Data sourced from https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
clear; 
clc;
close all;
format long g;

%% Import data for Linear Regression
data_table = readtable('Life_Expectancy_Data.csv');

T_keep = data_table.Year==2015;
data_table = data_table(T_keep,:);
%disp(data_table)

%% Remove columns with high number of missing values
disp(sum(ismissing(data_table))) % cols 3, 10, 13 & 14 contain lots of NaNs
data_table(:, {'Alcohol','TotalExpenditure','GDP','Population'}) = []; % remove these from dataset
data_table(:, 'percentageExpenditure') = []; % col  mostly 0s, as it's dependent on col Total Exp
disp(sum(ismissing(data_table)))

%% Deal with remaining NaNs by setting them to to mean of col with matching development status
for i = 1:length(data_table.Properties.VariableNames)
    for j = 4:length(data_table.Year)
        status = data_table.Status(j);
        if ismissing(data_table.(i)(j))
            data_table.(i)(j) = mean(data_table.(i)(~isnan(data_table.(i)) & matches(data_table.Status, status)));
        end
    end
end
disp(sum(ismissing(data_table)))

%% Preparing data for model selection
pred_data = data_table(:,5:end);
life_expectancy = data_table(:,4);
% Want to reduce number of variables in model so remove highly correlated vars
% Create a heatmap of variable correlations
r = corrcoef(pred_data.Variables);
% Replace upper triangle with NaNs
isupper = logical(triu(ones(size(r)),1));
r(isupper) = NaN;
h = heatmap(r,'MissingDataColor','w');
labels = pred_data.Properties.VariableNames;
h.XDisplayLabels = labels;
h.YDisplayLabels = labels; 
% Remove some of the variables which have highest correlation with other predictor variables
pred_data.IncomeCompositionOfResources = [];
pred_data.thinness5_9Years = [];
pred_data.Measles = [];
pred_data.under_fiveDeaths = [];
pred_data.Diphtheria = [];

%% Use forward selection based on correlation with life expectancy to select model
M = [[1:8]' corr(pred_data.Variables, life_expectancy.Variables)];
M = sortrows(M,2, "descend", ComparisonMethod="abs");

i=1; exit_cond = 0;
BIC_ind = zeros(1,length(M));
train_data = pred_data.(M(1,1));

while exit_cond == 0
    [~, ~, ~, BIC_ind(i)] = OLS(train_data, life_expectancy.Variables);
    if i==1 || BIC_ind(i) <= BIC_ind(i-1)
        i= i+1;
        train_data = [train_data, pred_data.(M(i,1))];
        continue
    end
    exit_cond = 1;
end

%% Display BIC output from forward selection model
BIC_out = [M(:,1)'; BIC_ind];
disp(BIC_out(:,BIC_ind~=0))
% We see that the 8th, 1st, 6th & 5th variables should all be included

%% Fit model based using first 4 predictor variables
final_data = pred_data(:,{'Schooling', 'AdultMortality', 'HIV_AIDS', 'Polio'});
[B, conf_int, t_p_val, ~, R2, R2_adj, F_test, DW_test, BP_test, JB_test, VIF] = OLS(final_data.Variables, life_expectancy.Variables);

%% OLS Estimator Output a)-c)
CoefNames = ['Constant', final_data.Properties.VariableNames];
coef_out = array2table([B, conf_int, t_p_val], 'VariableNames', {'Beta', 'Lower Bound 95% CI', 'Upper Bound 95% CI', 'p-value'},'RowNames',CoefNames);
disp(coef_out)

%% R-Squared Output d)
R2_out = array2table([R2, R2_adj], 'VariableNames',{'R-Squared','Adjusted R-Squared'});
disp(R2_out)

%% Residual Analysis Output e) & g)
res_out = array2table([F_test; DW_test; BP_test; JB_test], 'VariableNames',{'Test Statistic', 'Critical Value*', 'p-value'}, ...
    'RowNames',{'F-Statistic', 'Durbin-Watson', 'Breusch-Pagan', 'Jacque-Bera'});
res_out = convertvars(res_out, @isnumeric, @nanblank);

disp(res_out)

%% Multicollinearity Output h)
disp(VIF)
