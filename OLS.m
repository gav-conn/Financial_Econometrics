function [B, conf_int, t_p_val, BIC, R2, R2_adj, F_test, DW_test, BP_test, JB_test, VIF] = OLS(X, Y)

% Add column of ones to account for constant term
[T,K] = size(X);
X = [ones(T,1) X];

% Calculate Beta values of our OLS model
B = (X.'*X)\(X.'*Y);
% Calculate fitted Y values based on our regression
Y_pred = X*B;

% Calculate residuals to get standard error of estimates
e = Y-Y_pred;
s2 = (e.'*e)/(T-K);
var_B = s2*inv(X.'*X);
SE_B = sqrt(diag(var_B));
% Use standard error term to calculate confidence interval
conf_int = [B-1.96*SE_B, B+1.96*SE_B];

% Calculate BIC to help with model selection
BIC = T*log(sum(e.^2)/T)+K*log(T);

% Calculate t-statistic for each coefficient w/ associated p-value
tStat = abs(B./SE_B);
t_p_val = tcdf(tStat, T-K,"upper")*2;

% Calculate R-squared & Adjusted R-squared values of the model
Y_bar = mean(Y);
R2 = 1-sum(e.^2)/sum((Y-Y_bar).^2);
R2_adj = 1-((T-1)/(T-K)*(1-R2));

% Calculate F-statistic
F = (R2/K)/((1-R2)/(T-K));
F_crit = finv(0.95, K, T-K);
F_p_val = fcdf(F, K, T-K);
F_test = [F F_crit F_p_val];

clf
for i = 1:K+1
    % Plot regression line leaving all other variables equal to mean
    X_plot=[repmat(mean(X(:,1:i-1)),T,1), X(:,i) , repmat(mean(X(:,i+1:end)),T,1)];
    hold on
    scatter(X(:,i),Y)
    plot(X(:,i), X_plot*B)
    hold off
    if i < K+1
        nexttile
    end
end

% Use Jarque-Bera test to examine normaility of residuals
JB = (T-K)/6*(skewness(e)^2+(kurtosis(e)-3)^2/4);
JB_crit = chi2inv(0.95, 2);
JB_p_val = chi2cdf(JB, 2);
JB_test = [JB JB_crit JB_p_val];

% Use Breusch-Pagan test to examine heteroskedasticity of residuals
B_BP = (X.'*X)\(X.'*e.^2);
BP_pred = X*B_BP;
BP_bar = mean(e.^2);
R2_BP = sum((BP_pred-BP_bar).^2)/sum((e.^2-BP_bar).^2);
BP = R2_BP*T;
BP_crit = chi2inv(0.95, K);
BP_p_val = chi2cdf(BP, K);
BP_test = [BP BP_crit BP_p_val];

% Use Durbin-Watson test to examine serial autocorrelation of residuals
DW = sum(diff(e).^2)/sum(e.^2);
%DW_p_val = pvaluedw(DW, X, 'approximate');
DW_test = [DW missing missing]; % Don't have single critical value for DW-test 

% Test for multicollinearity by calculating Variance Inflation Factors for each of the regressors
R0 = corrcoef(X(:,2:end));
VIF = diag(inv(R0))';
VIF = array2table(VIF);
end
