% gennerate correlation matrix
% MSC8001
% Author: John Worrall
% Description:Given input data, create a ANFIS
% Requirments: 
%----------------------------

clear 
clc
close all

%Raw values --------------------------------
closeData=xlsread('Indexs.xlsx','AlIndex');
day = closeData(:, 1);
AORD = closeData(:, 2);
DJI = closeData(:, 3);
FTSE = closeData(:, 4);
HSI = closeData(:, 5);
N225 = closeData(:, 6);

A = [AORD DJI FTSE HSI N225];
A(any(isnan(A), 2), :) = []; %Remove Rows with NAN values

corrcoef(A) % Correlation matrix

corrplot(A) % Correlation plot
set(gca,'XTickLabel',['AORD' 'DJI' 'FTSE' 'HSI' 'N225']);
