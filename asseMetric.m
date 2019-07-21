function [r,ENS,d,Leg,RMSE,rRMSE,MAE,MAPE,PI]=asseMetric(dataObs,dataSim)
%Function requiring forcasted, observed values to then
%calculate acceptable assesment metrics. 

% correlation coefficient, r:
R21=sum((dataObs -mean(dataObs)).*( dataSim - mean(dataSim)));
R22=sqrt(sumsqr(dataObs -mean(dataObs)).*sumsqr(dataSim - mean(dataSim)));
r = R21/R22;

% Nash–Sutcliffe coefficient, ENS
ENS = (1-(sumsqr(dataObs - dataSim)/sumsqr(dataObs - mean(dataObs))));

% Willmottr’s Index, d
d =(1-(sumsqr(dataObs - dataSim)/sumsqr(abs(dataSim - mean(dataObs) + abs(dataObs -mean(dataObs))))));

% % **** NOT USED ****
% % Peak Percentage Deviation, Pdv 
% Pdv = (1-max(dataSim)/max(dataObs))*100;

% Root Mean Square Error, RMSE
RMSE = sqrt(sumsqr(dataObs - dataSim)/length(dataObs));

%Relative RMSE
rRMSE = (sqrt(sumsqr(dataObs - dataSim)/length(dataObs)))/(sum(dataObs)/length(dataObs)) * 100;

% Mean Absolute Error, MAE
MAE = (sum(abs(dataSim - dataObs)))/length(dataObs);

%MAPE
MAPE =(sum(abs((dataSim - dataObs)./dataObs)))/length(dataObs)*100;

%Legates & McCabes Index, Leg
L = sum(abs(dataSim - dataObs));
M = sum(abs(dataObs - mean(dataObs)));
Leg = (1-L/M);

% All error values within the same matrix, PI
%PI = [r ENS d Leg RMSE rRMSE MAE rMAE] ;
PI = [r ENS d Leg RMSE rRMSE MAE MAPE] ;

end