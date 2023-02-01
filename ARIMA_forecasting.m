%% Initial importation of Data
clearvars()

% Set import options - Import only columns under consideration in our analysis
opts = detectImportOptions('current.csv');
opts = setvaropts(opts,'sasdate','InputFormat','MM/dd/uuuu');
opts.SelectedVariableNames = {'sasdate','TB6MS'};

[data] = readtable('current.csv', opts);
data(1,:) = []; % Remove TCode information from dataframe

Date = table2array(data(:,1));
TBill = table2array(data(:,2));

%% Plot time-series of historical 6-Month Treasury Bill Rate 
plot(Date, TBill)

%% Plot sample ACF & sample PACF of 6-Month Treasury Bill Rate 
tiledlayout(2,1)
nexttile
autocorr(TBill)
nexttile
parcorr(TBill)

%% Check for integration using ADF test
[h,pValue] = adftest(TBill);
dTBill = diff(TBill);

%% Plot differenced series
clf
plot(Date(1:end-1), dTBill)

%% Plot sample ACF & sample PACF of differenced 6-Month Treasury Bill Rate 
tiledlayout(2,1)
nexttile
autocorr(dTBill)
nexttile
parcorr(dTBill)

%% Fit model based on previous 20 years of data
TBill_AIC_20Y=zeros(4,4);
TBill_BIC_20Y=zeros(4,4);

options =  optimset('lsqnonlin');
options.Display='none';

for p=0:3
    for q=0:3
    [~, ~, ~, ~, diagnostics] = armaxfilter(dTBill(end-240:end),1,[1:p],[1:q],[],[],options);
    TBill_AIC_20Y(p+1,q+1)=diagnostics.AIC; % store AIC values
    TBill_BIC_20Y(p+1,q+1) = diagnostics.SBIC; % store BIC values
    end
end

%% Examine AIC/BIC values of each of the estimated models for 20-Year data
% As BIC better penalizes model overfitting vs AIC, I feel it is the better model criterion in this instance
% For this reason our model selection will primarily be based off of this value
% New figure
figure()
% Use surf to produce a surface plot
surf(0:3,0:3,TBill_BIC_20Y)
% Labels - X is the first (columns of data), Y is the second (rows)
xlabel('MA Order')
ylabel('AR Order')
% Title
title('BIC')
[BIC_AR,BIC_MA] = find(TBill_BIC_20Y==min(min(TBill_BIC_20Y)));
disp('BIC Minimum at:')
disp([BIC_AR - 1 BIC_MA - 1]);
% One of ARMA(1,1) or ARMA(1,2) best based off of minimisation our BIC values
[AIC_AR,AIC_MA] = find(TBill_AIC_20Y==min(min(TBill_AIC_20Y)));
disp('AIC Minimum at:')
disp([AIC_AR - 1 AIC_MA - 1]);
% ARMA(3,1) minimises our AIC

% Based on the above diagnostics, I feel the models we should consider further are:
% ARMA(1,1), ARMA(1,2) & an ARMA(3,1)

%% Examine predictive performance on previous data to determine best model to use.
% In the following, we will examine the out-of-sample forecast performance of the model
% Use a forecast horizon of 3 years (36 months)
forecast_horizon = 36;
OOS_iter = length(dTBill)-forecast_horizon;
sample_Data = dTBill(OOS_iter-240:end);
n = length(sample_Data)-forecast_horizon;

% Estimate models based on 20 years of data, excluding most recent year
ToEstMdl_arma11 = arima('ARLags',1,'MALags',1);
[EstMdl_arma11, logL1] = estimate(ToEstMdl_arma11,sample_Data(1:end-forecast_horizon));

ToEstMdl_arma21 = arima('ARLags',1,'MALags',[1:2]);
EstMdl_arma21 = estimate(ToEstMdl_arma21,sample_Data(1:end-forecast_horizon));

ToEstMdl_arma31 = arima('ARLags',[1:3],'MALags',1);
EstMdl_arma31 = estimate(ToEstMdl_arma31,sample_Data(1:end-forecast_horizon));

%% Calculate the forecasts for each model.

[yhat_arma11, yMSE_arma11] = forecast(EstMdl_arma11,forecast_horizon,'Y0',sample_Data(1:end-forecast_horizon));
[yhat_arma12, yMSE_arma12] = forecast(EstMdl_arma21,forecast_horizon,'Y0',sample_Data(1:end-forecast_horizon));
[yhat_arma31, yMSE_arma31] = forecast(EstMdl_arma31,forecast_horizon,'Y0',sample_Data(1:end-forecast_horizon));


%% Plot forecasts for each model
t1 = datetime(Date(OOS_iter-240));
t2 = datetime(Date(OOS_iter+forecast_horizon));
fore_Dates = [t1:calmonths(1):t2]';

clf

hold on
plot(fore_Dates, sample_Data);
plot(fore_Dates,[repmat(missing,1, n) yhat_arma11']);
plot(fore_Dates,[repmat(missing,1, n) yhat_arma12']);
plot(fore_Dates,[repmat(missing,1, n) yhat_arma31']);
title('6-Month TBill Rate Forecast')
xlabel('Time')
ylabel('TBill Rate')
legend('Historical','ARMA(1,1)','ARMA(1,2)', 'ARMA(3,1)', Location='southwest')
grid on
hold off

%%
% calculate forecast error
EF_arma11 = dTBill(n+1:n+forecast_horizon)-yhat_arma11;
EF_arma12 = dTBill(n+1:n+forecast_horizon)-yhat_arma12;
EF_arma31 = dTBill(n+1:n+forecast_horizon)-yhat_arma31;
% calculate Root Mean Square Forecast Error
EF2_arma11=EF_arma11.^2;
EF2_arma12=EF_arma12.^2;
EF2_arma31=EF_arma31.^2;

RMSFE_arma11 = [sum(EF2_arma11)]/forecast_horizon;
RMSFE_arma12 = [sum(EF2_arma12)]/forecast_horizon;
RMSFE_arma31 = [sum(EF2_arma31)]/forecast_horizon;

disp([RMSFE_arma11 RMSFE_arma12 RMSFE_arma31]) % ARMA(2,1) minimises RMSFE

%% Diebold-Mariano test
[DM_1, prob_1] = dmtest1(EF_arma11, EF_arma12, forecast_horizon);
[DM_2, prob_2] = dmtest1(EF_arma11, EF_arma31, forecast_horizon);
[DM_3, prob_3] = dmtest1(EF_arma12, EF_arma31, forecast_horizon);

disp([DM_1 prob_1; DM_2 prob_2; DM_3 prob_3])
% p-val < 0.95 so can't reject null of same predictive performance.
% As ARMA(1,1) has same predictive performance with fewer parameters, proceed with this model

%% Proceed with final prediction using ARMA(1,1)

ToEstMdlm_ARMA11 = arima('ARLags',1,'MALags',1);
EstMdlm_ARMA11 = estimate(ToEstMdlm_ARMA11, dTBill(end-240:end));
[yhat_arma11, yMSE_arma11] = forecast(EstMdlm_ARMA11,forecast_horizon,'Y0',dTBill(end-240:end));

%% Plot forecasts for differenced values
n = 240+forecast_horizon;
t1 = datetime(Date(end-240));
t2 = datetime(Date(end)+calmonths(forecast_horizon));
fore_Dates = [t1:calmonths(1):t2]';

clf

hold on
plot(fore_Dates, [dTBill(end-240:end)' repmat(missing, 1, forecast_horizon)]);
plot(fore_Dates, [repmat(missing,1, 240) dTBill(end) yhat_arma11']);
plot(fore_Dates, [repmat(missing,1, 241) (yhat_arma11 + 1.96*sqrt(yMSE_arma11))'],'r:','LineWidth',2);
plot(fore_Dates, [repmat(missing,1, 241) (yhat_arma11 - 1.96*sqrt(yMSE_arma11))'],'r:','LineWidth',2);
title('Differenced 6-Month TBill Rate Forecast')
xlabel('Time')
ylabel('TBill Rate')
legend('Historical','ARMA(1,1)','95% Confidence Interval', Location='southwest')
grid on
hold off

% Forecast shows high values reverting to the mean

%% Plot forecast for integrated series
TBill_hat = cumsum(yhat_arma11)+TBill(end);
TBill_MSE = cumsum(yMSE_arma11);

clf

hold on
plot(fore_Dates, [TBill(end-240:end)' repmat(missing, 1, forecast_horizon)]);
plot(fore_Dates, [repmat(missing,1, 240) TBill(end) TBill_hat']);
plot(fore_Dates, [repmat(missing,1, 241) (TBill_hat + 1.96*sqrt(TBill_MSE))'],'r:','LineWidth',2);
plot(fore_Dates, [repmat(missing,1, 241) (TBill_hat - 1.96*sqrt(TBill_MSE))'],'r:','LineWidth',2);
title('6-Month TBill Rate Forecast')
xlabel('Time')
ylabel('TBill Rate')
legend('Historical','ARMA(1,1)','95% Confidence Interval', Location='northwest')
grid on
hold off
