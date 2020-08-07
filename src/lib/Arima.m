% ARIMA
% MSC8001
% Author: John Worrall
% Description :ARIMA model
% Requirments: enter in parameters p d q
%----------------------------

clear 
clc
close all

% Mdl = arima(p,D,q) creates a nonseasonal linear time series model using 
% autoregressive degree p, differencing degree D, and moving average degree q.
% ARIMA(p,d,q) forecasting equation
% ARIMA(1,0,0) = first-order autoregressive model
% ARIMA(0,1,0) = random walk 
% ARIMA(1,1,0) = differenced first-order autoregressive model 
% ARIMA(0,1,1) without constant = simple exponential smoothing 
% ARIMA(0,1,1) with constant = simple exponential smoothing with growth 
% ARIMA(0,2,1) or (0,2,2) without constant = linear exponential smoothing 
% ARIMA(1,1,2) with constant = damped-trend linear exponential smoothing 

p = 3;
d = 1;
q = 3;


%randomise parameters, ensure the results are reproducible.
rand('twister', 123);

%Raw values --------------------------------
closeData=xlsread('Indexs.xlsx','AlIndex');
Day = closeData(:, 1);
AORD = closeData(:, 2);
DJI = closeData(:, 3);
FTSE = closeData(:, 4);
HSI = closeData(:, 5);
N225 = closeData(:, 6);
A = [AORD Day DJI FTSE HSI N225];
A(any(isnan(A), 2), :) = []; %Remove Rows with NAN values
AORD = A(:, 1);
DayA = A(:, 2);
A = A(:,[1 2]);
clearvars -except A AORD DayA p d q %Remove all variable except A

% Split and train dataset --------------------------------
nData = size(A);
pTrain=0.9; % 90% spilit
%nTrainData=round(pTrain*nData);
nTrainData=round(1776);
PERM = 1:nData; 
TrainInd =PERM(1:nTrainData);
TestInd = PERM(nTrainData+1:end);

TrainSet = A(TrainInd,:);
TestSet = A(TestInd,:);
T2 = length(TestSet);

% Difference and PCAF --------------------------------
%initial investigation
figure(1)
hold on
subplot(2,1,1);
plot(AORD);
title('AORD');
subplot(2,1,2)
autocorr(AORD,25);
xlabel('lag (in days)');
hold off

A_t1 = diff(AORD,1);
figure(2)
hold on
subplot(2,1,1);
plot(A_t1);
title('First Difference of  AORD');
subplot(2,1,2)
autocorr(A_t1,25);
xlabel('lag (in days)');
hold off




% % PACF From p q d paramaters evaluation --------------------------------
% y = TrainSet(:, 1);
% % selecting best ARIMA(p,0,q)
% max_p=5; max_q=5;
% crit=zeros(max_p,max_q); c=0;
% N=size(y,1);
% 
% 
% d = 0;
% for i=1:max_p
%     for j=1:max_q
%     Mdl = arima(i,d,j); % <- shorthand syntax, model with a constant
%     [EstMdl,EstParamCov,logL,info] = estimate(Mdl,y);
%     [aic,bic]=aicbic(logL,i+j,N);
%         if ((i==1) && (j==1)) || (bic<c)
%             c=bic; % <- change this line for AIC
%         end           
%             p=i;
%             q=j;
%     crit(i,j)=bic; % <- change this line for AIC
%     end
% end


% ARIMA 1 -------------------------------- Best BIC
p = 2;
d = 1;
q = 2;

Md1 = arima(p,d,q);
[est, ~, logL, info] = estimate(Md1, TrainSet(:, 1));
%[yF2, YMSE] = forecast(est, T2,'Y0',y);
[yF1, YMSE] = forecast(est, T2,'Y0',TestSet(:, 1));


dataObs = TestSet(:, 1);
dataSim1 = yF1;

% Error Metrics  ----------------
[nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnMAPE,nnPI]=asseMetric(dataObs,dataSim1);
ArimaErrors1  = nnPI;
asseMetricVis(dataObs,dataSim1,nnR,2,'ARIMA1 - Nonseasonal');

% ARIMA 2 --------------------------------Best AIC
p = 1;
d = 1;
q = 5;

Md2 = arima(p,d,q);
[est, ~, logL, info] = estimate(Md2, TrainSet(:, 1));
%[yF2, YMSE] = forecast(est, T2,'Y0',y);
[yF2, YMSE] = forecast(est, T2,'Y0',TestSet(:, 1));


dataSim2 = yF2;

% Error Metrics  ----------------
[nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnMAPE,nnPI]=asseMetric(dataObs,dataSim2);
ArimaErrors2  = nnPI;
asseMetricVis(dataObs,dataSim2,nnR,2,'ARIMA2 - Nonseasonal');


% ARIMA 3 --------------------------------Seasonality

y = TrainSet(:, 1);
y2 = TestSet(:, 1);
T = length(TrainSet(:, 1));

Md3 = arima('Constant',0,'D',1,'Seasonality',12,'MALags',1,'SMALags',12);
EstMdl = estimate(Md3,y);

dataSim3 = yF2;

%[yF,yMSE] = forecast(EstMdl,60,'Y0',y);
[yF,yMSE] = forecast(EstMdl,T2,'Y0',y);
upper = yF + 1.96*sqrt(yMSE);
lower = yF - 1.96*sqrt(yMSE);

% ARIMA 2  ----------------
Md2 = arima(p,d,q);
[est, ~, logL, info] = estimate(Md2, TrainSet(:, 1));
%[yF2, YMSE] = forecast(est, T2,'Y0',y);
[yF2, YMSE] = forecast(est, T2,'Y0',TestSet(:, 1));

%EstMd2 = estimate(Md2,y);
% figure
% hold on
% plot(yF2,'Color',[.75,.75,.75])
% hold off


% Visualisations  ----------------
figure
plot(y,'Color',[.75,.75,.75])
hold on
h1 = plot(T+1:T+T2,dataSim1,'g','LineWidth',1);
h2 = plot(T+1:T+T2,dataSim2,'b','LineWidth',1);
h3 = plot(T+1:T+T2,upper,'k--','LineWidth',1.5);
h4 = plot(T+1:T+T2,TestSet(:, 1));
plot(T+1:T+T2,lower,'k--','LineWidth',1.5);
%plot(T+1:T+T2,TestSet(:, 1));
%xlim([0,T+592])
title('Forecast and 95% Forecast Interval')
legend([h1,h2,h3,h4],'ARIMA1 Seasonality','ARIMA2 Nonseasonal (p,d,q)','95% Interval','Test Values','Location','NorthWest')
hold off
% 
% dataObs = TestSet(:, 1);
% dataSim = yF2;
% Error Metrics  ----------------
% [nnR,nnENS,nnD,nnPDEV,nnRMSE,nnMAE,nnPI]=asseMetric(dataObs,dataSim);
% ArimaErrors  = nnPI;
% asseMetricVis(dataObs,dataSim,nnR,2,'ARIMA - Nonseasonal');


% % Need to read  ----------------
% %Concept
% https://www.otexts.org/fpp/8/1
% https://www.quora.com/Whats-the-difference-between-ARMA-ARIMA-and-ARIMAX-in-laymans-terms-What-exactly-do-P-D-Q-mean-and-how-do-you-know-what-to-put-in-for-them-in-say-R-1-0-2-or-2-1-1
% https://coolstatsblog.com/2013/08/14/using-aic-to-test-arima-models-2/
% https://stackoverflow.com/questions/44006832/estimate-model-order-of-an-autoregressive-ar-model
% https://people.duke.edu/~rnau/411arim2.htm


% implement
% https://stackoverflow.com/questions/22754460/estimate-aic-bic-criteria-from-given-signal
% https://au.mathworks.com/matlabcentral/answers/314740-arima-bic-loop-including-zero
% https://warwick.ac.uk/fac/soc/economics/current/modules/rm/notes1/research_methods_matlab_2.pdf
% https://stackoverflow.com/questions/44006832/estimate-model-order-of-an-autoregressive-ar-model





% % Not liscensed ARMA Model  ----------------
% sys = armax(TrainSet(:, 1),[4 3]);
% %yp = predict(sys,TrainSet(:, 1),10);
% opt = forecastOptions('InitialCOndition','z');
% yF3 = forecast(sys,TrainSet(:, 1),T2,opt);
% 
% %t = z9.SamplingInstants;
% %t1 = t(1:50);
% plot(t1,past_data,'k',t1,yp,'*b')
% legend('Past Data','Predicted Data')
% 



% % Whole Dataset  ----------------
% 
% % ARIMA 1 --------------------------------
% y3 = A(:, 1);
% T3 = length(A);
% Md3 = arima('Constant',0,'D',1,'Seasonality',12,'MALags',1,'SMALags',12);
% EstMd3 = estimate(Md3,y3);
% 
% %[yF,yMSE] = forecast(EstMdl,60,'Y0',y);
% [yF3,yMSE3] = forecast(EstMd3,T3,'Y0',y3);
% 
% 
% % ARIMA 2  ----------------
% Md4 = arima(p,d,q);
% [est, ~, logL, info] = estimate(Md4, A(:, 1));
% [yF4, YMSE] = forecast(est, T3,'Y0',y3);
% 
% EstMd2 = estimate(Md2,y3);
% 
% 

