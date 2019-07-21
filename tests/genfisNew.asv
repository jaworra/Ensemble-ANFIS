% ANFIS Multiple Input, trial on membership functions
% MSC8001
% Author: John Worrall
% Description:Given input data, create a ANFIS - Multi feature(other indexes) no random selection
% Requirments: 
%----------------------------

close all force
clear 
clc
close all

% ModelRun users Choice options to run
% 1 = single run (all inputs)
% 2 = Ensemble
% 3 = SOM+ANFIS,
% 4 = single run (all features except t-1)
% 5 = single run (just t-1)
% 6 = Mamdani model
% 7 = Arima
% 8 = Comparision of models
% 9 = plot timeseries (initial)
ModelRun = 3;  

MemFun = 'sugeno';  % Only Sugerno argument for ANFIS not Madami

%randomise parameters, so historic runs have no influence
rand('state',0);

%Raw values --------------------------------
closeData=xlsread('Indexs.xlsx','AlIndex');
Day = closeData(:, 1);
AORD = closeData(:, 2);
DJI = closeData(:, 3);
FTSE = closeData(:, 4);
HSI = closeData(:, 5);
N225 = closeData(:, 6);
A = [AORD DJI FTSE HSI N225 Day];
A(any(isnan(A), 2), :) = []; %Remove Rows with NAN values
clearvars -except ModelRun MemFun A Day; %Remove all variable except A

%Prep features --------------------------------
% ouput delete first row
AORDOut = A(:, 1);
AORDOut(1,:)=[];
DayA = A(:, 6);
DayA(1,:)=[];
% t-1 delete last row
AORDIn = A(:, 1);
AORDIn(end,:)=[];
DJIIn = A(:, 2);
DJIIn(end,:)=[];
FTSEIn = A(:, 3);
FTSEIn(end,:)=[];
HSIIn = A(:, 4);
HSIIn(end,:)=[];
N225In = A(:, 5);
N225In(end,:)=[];

%Normalise values --------------------------------
%Min Max values
MaxAordOt = max(AORDOut);
MinAordOt = min(AORDOut);
MaxAordIn = max(AORDIn);
MinAordIn = min(AORDIn);
MaxDji = max(DJIIn);
MinDji = min(DJIIn);
MaxFtse = max(FTSEIn);
MinFtse = min(FTSEIn);
MaxHsi = max(HSIIn);
MinHsi = min(HSIIn);
MaxN225 = max(N225In);
MinN225 = min(N225In);

for x = 1:numel(AORDOut)
   AORD_out(x) = (AORDOut(x) - MinAordOt)/(MaxAordOt - MinAordOt);
   AORD_t1(x) = (AORDIn(x) - MinAordIn)/(MaxAordIn - MinAordIn);
   DJI_t1(x) = (DJIIn(x) - MinDji)/(MaxDji - MinDji);
   FTSE_t1(x) = (FTSEIn(x) - MinFtse)/(MaxFtse - MinFtse);
   HSI_t1(x) = (HSIIn(x) - MinHsi)/(MaxHsi - MinHsi);
   N225_t1(x) = (N225In(x) - MinN225)/(MaxN225 - MinN225);
end
AORD_out = AORD_out';
AORD_t1 = AORD_t1';
DJI_t1 = DJI_t1';
FTSE_t1 = FTSE_t1';
HSI_t1 = HSI_t1';
N225_t1 = N225_t1';

A_Final = [AORD_out AORD_t1 DJI_t1 FTSE_t1 HSI_t1 N225_t1 DayA];

%Correlation plot
% Index_norm = [AORD_out DJI_t1 FTSE_t1 HSI_t1 N225_t1];
% corrplot(Index_norm,'varNames',{'AORD','DJI','FTSE','HSI','N225'});
% %


% Split and train dataset --------------------------------
nData = size(A_Final,1);
NInput = [AORD_t1 DJI_t1 FTSE_t1 HSI_t1 N225_t1];
NOutput = AORD_out;

PERM = 1:nData; 
pTrain=0.8;  %80 train  
pValid=0.1;  %10 validate
pTest= 0.1;  %10 test 
%Apply in order
TrainInd=PERM(1:round(pTrain*nData));
ValidInd=PERM(round(pTrain*nData)+1 :round(pTrain*nData)+round(pValid*nData));
TestInd=PERM(round(pTrain*nData)+round(pValid*nData)+1 :end); 

TrainInputs=NInput(TrainInd,:);
TrainTargets=NOutput(TrainInd,:);
ValidInputs=NInput(ValidInd,:);
ValidTargets=NOutput(ValidInd,:);
TestInputs=NInput(TestInd,:);
TestTargets=NOutput(TestInd,:);

clearvars -except ModelRun MemFun A A_Final pTrain pTest pValid TrainInd ValidInd MaxAordOt MinAordOt TrainInputs TrainTargets ValidInputs ValidTargets TestInputs TestTargets

% Machine learning --------------------------------
if ModelRun == 1 % Single Run ANFIS

% training stage -------------------------------- 
    %Set up structure for anfis
    fis = genfis3(TrainInputs, TrainTargets, MemFun, 'auto', [])
    
%     %Plot Membership rules
%     for input_index=1:1
%         subplot(1,1,input_index)
%         [x,y]=plotmf(fis,'input',input_index);
%         plot(x,y)S
%         axis([-inf inf 0 1.2]);
%         xlabel(['Input' int2str(input_index)]);
%     end

%     %Plot training output
%     subplot(2,1,1)
%     x = 1:length(TrainInputs);
%     yp = evalfis(TrainInputs, fis);
%     plot(x,TrainTargets,x,yp);
%     title('ANFIS vs Training Data')
%     xlabel('Days');
%     ylabel('Normalised value');
%     legend('Training','ANFIS','Location','SouthEast')
%     %Training Errors
%     ypEr = yp - TrainTargets;
%     subplot(2,1,2)
%     plot(x,ypEr);
%     title('Training Data Errors')
%     xlabel('Days');
%     ylabel('Errors');
%     legend('Training Errors','Location','SouthEast')    
    
%     %GUI training output
%      neuroFuzzyDesigner(fis)
%      fuzzyLogicDesigner(fis)
%      surfview(fis)    
    

 % validation stage --------------------------------    
    %Configure setting for validation
    EpochNum = 30;
    opt = anfisOptions('InitialFIS',fis);
    opt.DisplayANFISInformation = 0;
    opt.DisplayErrorValues = 0;
    opt.DisplayStepSize = 0;
    opt.DisplayFinalResults = 0;
    opt.ValidationData = [ValidInputs ValidTargets];
    opt.EpochNumber = EpochNum;
    opt.StepSizeIncreaseRate = 2*opt.StepSizeIncreaseRate;

    %Run validation and predicion
    %out_fis = anfis([TrainInputs TrainTargets],opt);
    [out_fis,trn_error,step_size,chk_out_fismat,chk_error] = anfis([TrainInputs TrainTargets],opt);

    %Plot training output    
    x = 1:length(TestTargets);
    yp = evalfis(TestInputs, out_fis);
    plot(x,TestTargets,x,yp);
    
    %Plot step size
    plot(step_size);
    
    %plot training and validation errors
    figure(2)
    x = [1:EpochNum];
    plot(x,trn_error,'.b',x,chk_error,'*r');
    title('Training vs Validation Error')
    xlabel('Epoch');
    ylabel('Error (RMSE)');
    legend('Training','Validation','Location','NorthWest')

%     %GUI validation output    
%     neuroFuzzyDesigner(out_fis)
%     fuzzyLogicDesigner(out_fis)
%     surfview(out_fis)
    
    %Predict Train
    predictTrain = evalfis(TrainInputs, fis);
    for x = 1:numel(TestTargets)   
       dataSimTrain(x) = predictTrain(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataObsTrain(x) = TrainTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataSimTrain = dataSimTrain';
    dataObsTrain = dataObsTrain';
    
    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObsTrain,dataSimTrain);
    sugenoErrorsTrain = nnPI;
    asseMetricVis(dataObsTrain,dataSimTrain,nnR,2, strcat(MemFun, ' Train (Aords with all features)'));
  
 % test stage --------------------------------  
    %predict Test
    predict = evalfis(TestInputs, out_fis);   
    
    for x = 1:numel(TestTargets)
       dataObs(x) = TestTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataSim(x) = predict(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataObs = dataObs';
    dataSim = dataSim';

    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObs,dataSim);
    A_SugenoAllFeatErrors = nnPI;
    asseMetricVis(dataObs,dataSim,nnR,2, strcat(MemFun, ' (Aords with all features)'));

    
%---------------------------------------------------------------
elseif ModelRun == 2    % Enesemble ANFIS
%---------------------------------------------------------------
    nModels = 10;

    TrainSet = [TrainTargets TrainInputs];
    ValidSet = [ValidTargets ValidInputs];
    for x = 1:numel(TestTargets)
       dataObs(x) = TestTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataObs = dataObs';

    for x = 1:nModels
        clearvars dataSim predict TrainTargets TrainInputs ValidTargets ValidInputs 

        %Trainset and Validation set for random sample with replacement.
        TrainSamp = datasample(TrainSet,size(TrainSet,1));% returns k observations sampled uniformly at random, with replacement, from the data in data.
        TrainTargets =  TrainSamp(:, 1);
        TrainInputs = TrainSamp(:,[2 3 4 5 6]);
        ValidSamp = datasample(ValidSet,size(ValidSet,1));% returns k observations sampled uniformly at random, with replacement, from the data in data.
        ValidTargets =  ValidSamp(:, 1);
        ValidInputs = ValidSamp(:,[2 3 4 5 6]);

        %ANFIS
        fis = genfis3(TrainInputs, TrainTargets,'sugeno', 'auto', []); 
        
        %View Membership rules
%         for input_index=1:1
%             subplot(1,1,input_index)
%             [x1,y]=plotmf(fis,'input',input_index);
%             plot(x1,y)
%             axis([-inf inf 0 1.2]);
%             xlabel(['Input' int2str(input_index)]);
%         end

        %Configure validation
        opt = anfisOptions('InitialFIS',fis);
        opt.DisplayANFISInformation = 0;
        opt.DisplayErrorValues = 0;
        opt.DisplayStepSize = 0;
        opt.DisplayFinalResults = 0;
        opt.ValidationData = [ValidInputs ValidTargets];

        [out_fis,trn_error,step_size,chk_out_fismat,chk_error] = anfis([TrainInputs TrainTargets],opt);
        mfEpoch(:,x)= opt.EpochNumber; % Stores number of epochs
        [m,n2]= size(out_fis.rule);  % Stores number of mf
        mfNumbers(:,x)= n2;
        
        %Predict Train
        predictTrain = evalfis(TrainInputs, fis);
        for z = 1:numel(TrainTargets)   
           dataSimTrain(z) = predictTrain(z) * (MaxAordOt - MinAordOt)+ MinAordOt;
           dataObsTrain(z) = TrainTargets(z) * (MaxAordOt - MinAordOt)+ MinAordOt;
        end
        dataSimTrain = dataSimTrain';
        dataObsTrain = dataObsTrain';
        runsTrain(:,x) = dataSimTrain;

        [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObsTrain,dataSimTrain);
        runsTrainError(:,x) = nnPI;
        %asseMetricVis(dataObsTrain,dataSimTrain,nnR,2, strcat(MemFun, ' Train (Aords with all features)'));

        %Predict Train
        predict = evalfis(TestInputs, out_fis);

        % denormalise simulation values --------------------------------
        for y = 1:numel(predict)
            dataSim(y) = predict(y) * (MaxAordOt - MinAordOt)+ MinAordOt;
        end   
        dataSim = dataSim';
        runs(:,x) = dataSim;

        RunNumber = num2str(x);
        %Error metrics --------------------------------
        [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObs,dataSim);
        runsError(:,x) = nnPI;
        %asseMetricVis(dataObs,dataSim,nnR,2,strcat('Sugeno, Aords with all features - Run ',RunNumber));       

    end

    
    %Mean model Train
    avModelTrain = mean(runsTrain,2);
    avModelTrainErr = mean(runsTrainError,2)';
%     [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI]=asseMetric(dataObsTrain,avModelTrain);
%     avModelTrainErr = nnPI';
%     dataObsTrain = dataObsTrain';
%     asseMetricVis(dataObsTrain,avModelTrainErr,nnR,2,strcat('Sugeno, ENSEMBLE Train Aords with all features - Run ',RunNumber));
    
    
    %Mean model TEST
    avModel = mean(runs,2);
    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObs,avModel);
    A_EnsembleANFIS_TRAIN_Errors = nnPI';
    asseMetricVis(dataObs,avModel,nnR,2,strcat('Sugeno, ENSEMBLE Aords with all features - Run ',RunNumber));
    
        
    min(mfEpoch)
    max(mfEpoch)
    min(mfNumbers)
    max(mfNumbers)   
    
    asseMetricVis(dataObs,runs,nnR,3, strcat(MemFun, ' (Aords with all features)'));  
    
    %box plot 
    asseMetricVis(dataObs,runs,nnR,4, strcat(MemFun, ' (Aords with all features)'));  
    asseMetricVis(dataObs,avModel,nnR,5, strcat(MemFun, ' (Aords with all features)'));  
    

%---------------------------------------------------------------   
elseif ModelRun == 3    % SOM ANFIS
%---------------------------------------------------------------
%   Redo TrainSet to include dates and validation set
%   Use SOM to seperated training and validation set

    pTrainSOM = pTrain+pValid;  %e.g 0.8+0.1
    nData = size(A_Final,1);
    PERM = 1:nData; 
    TrainIndSOM=PERM(1:round(pTrainSOM*nData));

    TrainSetSom = A_Final(TrainIndSOM,:);   

    %SOM values --------------------------------
    AORD=TrainSetSom(:, 1:6)'; %all features as input as dimensions
    
    %Create Topologies
    nrow = 10;
    ncol = 10;
    pos = gridtop([nrow ncol]); %grid pattern
    plotsom(pos);
    %Distance between points
    D2 = dist(pos); 
    
    % %Split into two
    % half = length(AORD)/2;
    % s1 = AORD(1:half);
    % s2 = AORD(half + 1 : end);
    % AORD2 = [s1' s2'];
    %AORD2=AORD2';
    AORD2 = AORD;
     
    net.trainParam.epochs = 1000;
    net.property.iteration.number = 200; 
    
    net = selforgmap([nrow ncol]);
    net = train(net,AORD2); %Train
    disp(net)
    y = net(AORD2);%Test
    classes = vec2ind(y);
    disp(classes)
    
    %Visual  --------------------------------
    figure(1)
    view(net);
    plotsomnc(net);
    figure(2)
    plotsomnd(net);
    plotsomplanes(net);
    
    figure(3)
    plotsomhits(net,AORD2)
    figure(4);
    plotsompos(net,AORD2);
    
    %Extract clusters for input  --------------------------------
    hits=sum(sim(net,AORD2)');
    hits=hits';
    classes = vec2ind(net(AORD2))';
    AORD_cluster = [TrainSetSom classes]; %Add to main input
    
    %rearrane into cluster ascending --------------------------------
    B_Final = sortrows(AORD_cluster, 8); %Reset vector (includes cluster number and orginal ordered day)
    
    %--------------------------------
    %Hybrid SOOM-ANFIS
    %--------------------------------
         
    %TrainInd=PERM(1:round(pTrain*nData));
    %ValidInd=PERM(round(pTrain*nData)+1 :round(pTrain*nData)+round(pValid*nData));
    
    NOutputSOM =  B_Final(:,1); % SOM
    NInputSOM = B_Final(:,[2 3 4 5 6]); %SOM
    %NInputSOM = B_Final(:,[2 3 4 5 6 8]); %SOM with clusters
    
    %Split train and validate based on clusters from SOM
    TrainInputs=NInputSOM(TrainInd,:);
    TrainTargets=NOutputSOM(TrainInd,:);
    ValidInputs=NInputSOM(ValidInd,:);
    ValidTargets=NOutputSOM(ValidInd,:);

    %Set up structure for anfis
    fis = genfis3(TrainInputs, TrainTargets, MemFun, 'auto', []); 

    %View Membership rules
    for input_index=1:1
        subplot(1,1,input_index)
        [x,y]=plotmf(fis,'input',input_index);
        plot(x,y)
        axis([-inf inf 0 1.2]);
        xlabel(['Input' int2str(input_index)]);
    end
   

    %Configure validation
    opt = anfisOptions('InitialFIS',fis);
    opt.DisplayANFISInformation = 0;
    opt.DisplayErrorValues = 0;
    opt.DisplayStepSize = 0;
    opt.DisplayFinalResults = 0;
    opt.ValidationData = [ValidInputs ValidTargets];
    %opt.EpochNumber = 5;

    %Run validation and predicion
    %out_fis = anfis([TrainInputs TrainTargets],opt);
    [out_fis,trn_error,step_size,chk_out_fismat,chk_error] = anfis([TrainInputs TrainTargets],opt);
%      neuroFuzzyDesigner(out_fis)
%      fuzzyLogicDesigner(out_fis)
%      surfview(out_fis)
    
    %Predict Train
    predictTrain = evalfis(TrainInputs, fis);
    for x = 1:numel(TestTargets)   
       dataSimTrain(x) = predictTrain(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataObsTrain(x) = TrainTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataSimTrain = dataSimTrain';
    dataObsTrain = dataObsTrain';

    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnMAPE,nnPI] =asseMetric(dataObsTrain,dataSimTrain);
    SomANFIS_TRAIN_Errors = nnPI;
    asseMetricVis(dataObsTrain,dataSimTrain,nnR,2, strcat(MemFun, ' Train SOM-ANFIS (Aords with all features)'));

    %Predict Test
    predict = evalfis(TestInputs, out_fis);

   for x = 1:numel(TestTargets)
       dataObs(x) = TestTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataSim(x) = predict(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataObs = dataObs';
    dataSim = dataSim';

    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObs,dataSim);
    A_SomANFIS_TEST_Errors = nnPI;
    asseMetricVis(dataObs,dataSim,nnR,2, strcat(MemFun, ' SOM-ANFIS (Aords with all features)'));

    
%---------------------------------------------------------------   
elseif ModelRun == 4    % single run surgeno (all features except t-1)
%--------------------------------------------------------------
    TrainInputs = TrainInputs(:,[2 3 4 5]);
    ValidInputs = ValidInputs(:,[2 3 4 5]);
    TestInputs = TestInputs(:,[2 3 4 5]); 
    
    
    %Set up structure for anfis
    fis = genfis3(TrainInputs, TrainTargets, MemFun, 'auto', []); 
    neuroFuzzyDesigner(fis)
    fuzzyLogicDesigner(fis)
    surfview(fis)
    %View Membership rules
    % for input_index=1:1
    %     subplot(1,1,input_index)
    %     [x,y]=plotmf(fis,'input',input_index);
    %     plot(x,y)
    %     axis([-inf inf 0 1.2]);
    %     xlabel(['Input' int2str(input_index)]);
    % end

    %Configure validation
    opt = anfisOptions('InitialFIS',fis);
    opt.DisplayANFISInformation = 0;
    opt.DisplayErrorValues = 0;
    opt.DisplayStepSize = 0;
    opt.DisplayFinalResults = 0;
    opt.ValidationData = [ValidInputs ValidTargets];
    %opt.EpochNumber = 5;
    
 

    %Run validation and predicion
    %out_fis = anfis([TrainInputs TrainTargets],opt);
    [out_fis,trn_error,step_size,chk_out_fismat,chk_error] = anfis([TrainInputs TrainTargets],opt);
     neuroFuzzyDesigner(out_fis)
     fuzzyLogicDesigner(out_fis)
     surfview(out_fis)
    
    predictTrain = evalfis(TrainInputs, fis);
    for x = 1:numel(TestTargets) %SHOULD THIS BE TrainTargest INSTEAD of TestTargets  
       dataSimTrain(x) = predictTrain(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataObsTrain(x) = TrainTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataSimTrain = dataSimTrain';
    dataObsTrain = dataObsTrain';
    
    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObsTrain,dataSimTrain);
    sugenoTrain_exceptTminus1_Errors = nnPI;
    asseMetricVis(dataObsTrain,dataSimTrain,nnR,2, strcat(MemFun, ' Train (Aords with all features)'));

    
  
    predict = evalfis(TestInputs, out_fis);
    %plot
    % [a,b] = min(chk_error);
    % plot(1:5,trn_error,'g-',1:5,chk_error,'r-',b,a,'ko')
    % title('Training (green) and checking (red) error curve','fontsize',10)
    % xlabel('Epoch numbers','fontsize',10)
    % ylabel('RMS errors','fontsize',10)


    for x = 1:numel(TestTargets)
       dataObs(x) = TestTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataSim(x) = predict(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataObs = dataObs';
    dataSim = dataSim';

    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObs,dataSim);
    A_SugenoWithoutminus1Errors = nnPI;
    asseMetricVis(dataObs,dataSim,nnR,2, strcat(MemFun, ' (Aords without t-1)'));
    
%---------------------------------------------------------------   
elseif ModelRun == 5   % single run surgeno (just t-1)
%--------------------------------------------------------------
    TrainInputs = TrainInputs(:,[1]);
    ValidInputs = ValidInputs(:,[1]);
    TestInputs = TestInputs(:,[1]); 
    
   
    %Set up structure for anfis
    fis = genfis3(TrainInputs, TrainTargets, MemFun, 'auto', []); 
    gensurf(fis)
    
    %View Membership rules
    for input_index=1:1
        subplot(1,1,input_index)
        [x,y]=plotmf(fis,'input',input_index);
        plot(x,y)
        axis([-inf inf 0 1.2]);
        xlabel(['Input' int2str(input_index)]);
    end

    %Configure validation
    opt = anfisOptions('InitialFIS',fis);
    opt.DisplayANFISInformation = 0;
    opt.DisplayErrorValues = 0;
    opt.DisplayStepSize = 0;
    opt.DisplayFinalResults = 0;
    opt.ValidationData = [ValidInputs ValidTargets];
    %opt.EpochNumber = 5;

   
    %Run validation and predicion
    %out_fis = anfis([TrainInputs TrainTargets],opt);
    [out_fis,trn_error,step_size,chk_out_fismat,chk_error] = anfis([TrainInputs TrainTargets],opt)

    %Predict Train
    predictTrain = evalfis(TrainInputs, fis);
    for x = 1:numel(TestTargets)   
       dataSimTrain(x) = predictTrain(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataObsTrain(x) = TrainTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataSimTrain = dataSimTrain';
    dataObsTrain = dataObsTrain';
    
    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObsTrain,dataSimTrain);
    sugenoTrain_withTminus1_Errors = nnPI;
    asseMetricVis(dataObsTrain,dataSimTrain,nnR,2, strcat(MemFun, ' Train (Aords with all features)'));

    
    %Predict Test
    predict = evalfis(TestInputs, out_fis);
    for x = 1:numel(TestTargets)
       dataObs(x) = TestTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataSim(x) = predict(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataObs = dataObs';
    dataSim = dataSim';

    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObs,dataSim);
    A_sugenoTEST_withTminus1_Errors = nnPI;
    asseMetricVis(dataObs,dataSim,nnR,2, strcat(MemFun, ' (Aords with just t-1)'));

%---------------------------------------------------------------   
elseif ModelRun == 6   % single run madami (just t-1)
%--------------------------------------------------------------
    TrainInputs = TrainInputs(:,[1]);
    ValidInputs = ValidInputs(:,[1]);
    TestInputs = TestInputs(:,[1]); 
    

    %As madami is not accepted in ANFIS -> apply test set (Test + Validation)
    TrainInputs = [TrainInputs;ValidInputs];
    TrainTargets = [TrainTargets;ValidTargets];


    fis = genfis3(TrainInputs, TrainTargets, 'mamdani', 'auto', []);
    predict = evalfis(TestInputs, fis);
    
    for x = 1:numel(TestTargets)
       dataObs(x) = TestTargets(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
       dataSim(x) = predict(x) * (MaxAordOt - MinAordOt)+ MinAordOt;
    end
    dataObs = dataObs';
    dataSim = dataSim';
    
    [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObs,dataSim);
    A_Mamdani_Errors = nnPI;
    asseMetricVis(dataObs,dataSim,nnR,2, strcat(MemFun, ' (Aords without t-1)'));

    
    
%---------------------------------------------------------------   
elseif ModelRun == 7   % ARIMA
%-------------------------------------------------------------- 
    
    
    %Done in R
    
%---------------------------------------------------------------   
elseif ModelRun == 8   % comparison of modesl
%--------------------------------------------------------------
    clear 
    clc
    close all
    
    reData=xlsread('ResultsRun.xlsx','Results_FINAL');
    test = reData(:, 1);
    arima1 = reData(:, 2);
    arima2 = reData(:, 3);
    t1mamdami = reData(:, 4);
    t1Sugerno = reData(:, 5);
    allSugerno = reData(:, 6);
    nt1Sugerno = reData(:, 7);
    en50ANFIS = reData(:, 8);
    en100ANFIS = reData(:, 9);  
    SOMANFIS = reData(:, 10); 
    arima3 = reData(:, 11);
    arima4 = reData(:, 12);
    arima5 = reData(:, 13);
    arima6 = reData(:, 14);
    arima7 = reData(:, 15);
    arima8 = reData(:, 16);
    arima9 = reData(:, 17);
    en5ANFIS = reData(:, 18);
    en10ANFIS = reData(:, 19);  
    
    models = reData(:,[1 2 3 4 5 6 7 8 9 10]);
     
%--------------------------------------------------------------------------------- 
%   %Boxplot for ARMIA results 1
%     figure
%     models = reData(:,[11 13 14 12 15]);
%     arima3E = abs(test - arima3);
%     arima5E = abs(test - arima5);
%     arima6E = abs(test - arima6);
%     arima4E = abs(test - arima4);
%     arima7E = abs(test - arima7);
%     
%     
%     models = [arima3E,arima5E,arima6E,arima4E,arima7E];
%     %boxplot(models)
%     boxplot(models,'symbol','','PlotStyle','traditional','BoxStyle','outline','Colors','k','Notch','off','MedianStyle','line','labels',{'ARIMA(0,1,0)','ARIMA(0,1,1)','ARIMA(2,1,0)','ARIMA(1,1,0)','ARIMA Seasonal(0,1,0)(0,0,1)'});
%     ylabel('forecasted error')
%     %xlabel('t1Sugerno','allSugerno','ARIMA 010')
%     title('Model Error Distribution');
%     h = findobj(gca,'Tag','Box');
%     for j=1:length(h)
%         patch(get(h(j),'XData'),get(h(j),'YData'),'y','FaceAlpha',.5);
%     end  
%     
%--------------------------------------------------------------------------------- 
% %   scatter plot for ANFIS Results 1
%     subplot(2,2,1)
%     hold on;
%     scatter(test,t1Sugerno)
%     title('Sugeno (t-1 All Ords)');
%     xlabel('Data Observation (Index)');
%     ylabel('Data Simulated (Index)');
%     %add line
%     nnR = corrcoef(test,t1Sugerno);
%     nnR = nnR(1,2);
%     coeffs = polyfit(test, t1Sugerno, 1);
%     % Get fitted values
%     fittedX = linspace(min(test), max(test), 198);
%     fittedY = polyval(coeffs, fittedX);
%     C = round(polyval(coeffs, fittedX(1,1),0));
%     plot(fittedX, fittedY, 'r-', 'LineWidth', 1);
%     txt1 = ['y = ' num2str(round(nnR,3)) 'x + ' num2str(C)];
%     text(min(test),max(t1Sugerno), txt1);
%     text(min(test),max(t1Sugerno)-50, ['r^2 = ' num2str(round(nnR*nnR,3))]); 
%     hold off;
%     subplot(2,2,2)
%     hold on;
%     scatter(test,allSugerno)
%     title('Sugeno (all Indexes t-1)');
%     xlabel('Data Observation (Index)');
%     ylabel('Data Simulated (Index)');
%     %add line
%     nnR = corrcoef(test,allSugerno)
%     nnR = nnR(1,2);
%     coeffs = polyfit(test, allSugerno, 1);
%     % Get fitted values
%     fittedX = linspace(min(test), max(test), 198);
%     fittedY = polyval(coeffs, fittedX);
%     C = round(polyval(coeffs, fittedX(1,1),0));
%     plot(fittedX, fittedY, 'r-', 'LineWidth', 1);
%     txt1 = ['y = ' num2str(round(nnR,3)) 'x + ' num2str(C)];
%     ylim =get(gca,'ylim')-30 
%     text(min(test),max(ylim), txt1);
%     text(min(test),max(ylim)-50, ['r^2 = ' num2str(round(nnR*nnR,3))]);  
%     hold off;
%     subplot(2,2,3)
%     hold on;
%     scatter(test,nt1Sugerno)
%     title('Sugeno (all Indexes without All Ords t-1)');
%     xlabel('Data Observation (Index)');
%     ylabel('Data Simulated (Index)');
%     %add line
%     nnR = corrcoef(test,nt1Sugerno)
%     nnR = nnR(1,2);
%     coeffs = polyfit(test, nt1Sugerno, 1);
%     % Get fitted values
%     fittedX = linspace(min(test), max(test), 198);
%     fittedY = polyval(coeffs, fittedX);
%     C = round(polyval(coeffs, fittedX(1,1),0));
%     plot(fittedX, fittedY, 'r-', 'LineWidth', 1);
%     txt1 = ['y = ' num2str(round(nnR,3)) 'x + ' num2str(C)];
%     ylim =get(gca,'ylim')-30 
%     text(min(test),max(ylim), txt1);
%     text(min(test),max(ylim)-70, ['r^2 = ' num2str(round(nnR*nnR,3))]);  
%     hold off;
%     subplot(2,2,4)
%     hold on;
%     scatter(test,t1mamdami)
%     title('Mamdani (t-1 All Ords )');
%     xlabel('Data Observation (Index)');
%     ylabel('Data Simulated (Index)');
%     %add line
%     nnR = corrcoef(test,t1mamdami)
%     nnR = nnR(1,2);
%     coeffs = polyfit(test, t1mamdami, 1);
%     % Get fitted values
%     fittedX = linspace(min(test), max(test), 198);
%     fittedY = polyval(coeffs, fittedX);
%     C = round(polyval(coeffs, fittedX(1,1),0));
%     plot(fittedX, fittedY, 'r-', 'LineWidth', 1);
%     txt1 = ['y = ' num2str(round(nnR,3)) 'x + ' num2str(C)];
%     ylim =get(gca,'ylim')+80 
%     text(min(test),min(t1mamdami)+5, txt1);
%     text(min(test),min(t1mamdami)+1, ['r^2 = ' num2str(round(nnR*nnR,3))]);  
%     hold off;
    
    
 
%  %Results 1  
% Timeseries
%     figure
%     plot(test,'k');
%     hold on;
%     plot(t1Sugerno);
%     plot(allSugerno);
%     plot(arima3);
%     title('ANFIS vs ARIMA')
%     xlabel('days')
%     ylabel('\Delta [All Ord Price]')
%     var = {{'Observation','t1Sugerno','allSugerno','ARIMA 010'},'Location','northwest'}
%     legend(var{:})
%     %legend({'Test','arima1','arima2','t1mamdami','t1Sugerno','allSugerno','nt1Sugerno','en50ANFIS','en100ANFIS','SOMANFIS'},'Location','northwest','NumColumns',2)
%     hold off

%Boxplot
%     figure
%     models = reData(:,[1 5 6 11]);
%     t1SugernoE = abs(test - t1Sugerno);
%     allSugernoE = abs(test - allSugerno);
%     arima3E = abs(test - arima3);
%     models = [t1SugernoE,allSugernoE,arima3E];
%     %boxplot(models)
%     boxplot(models,'symbol','','PlotStyle','traditional','BoxStyle','outline','Colors','k','Notch','off','MedianStyle','line','labels',{'SingleFeat(t-1) Sugerno','AllFeat(t-1) Sugerno','ARIMA 010'});
%     ylabel('forecasted error')
%     %xlabel('t1Sugerno','allSugerno','ARIMA 010')
%     title('Model Error Distribution');
%     h = findobj(gca,'Tag','Box');
%     for j=1:length(h)
%         patch(get(h(j),'XData'),get(h(j),'YData'),'y','FaceAlpha',.5);
%     end
    
    
    
 %Results 2   
% time series
%     plot(test,'k');
%     hold on;
%     plot(SOMANFIS);
%     plot(en100ANFIS);
%     title('SOM-ANFIS vs Ensemble ANFIS')
%     xlabel('days')
%     ylabel('\Delta [All Ord Price]')
%     var = {{'Observation','SOM-ANFIS','Ensemble100 ANFIS'},'Location','southwest'}
%     legend(var{:})
%     %legend({'Test','arima1','arima2','t1mamdami','t1Sugerno','allSugerno','nt1Sugerno','en50ANFIS','en100ANFIS','SOMANFIS'},'Location','northwest','NumColumns',2)
%     hold off

%  
%  % boxplot
%     %models = reData(:,[1 8 9 10]);
%     en50ANFISE = abs(test - en50ANFIS);
%     en100ANFISE = abs(test - en100ANFIS);
%     SOMANFISE = abs(test - SOMANFIS);
%     models = [SOMANFISE,en100ANFISE,en50ANFISE];
%     %boxplot(models)
%     boxplot(models,'symbol','','PlotStyle','traditional','BoxStyle','outline','Colors','k','Notch','off','MedianStyle','line','labels',{'SOM-ANFIS','Ensemble100 ANFIS','Ensemble50 ANFIS'});
%     ylabel('forecasted error')
%     title('Model Error Distribution');
%     h = findobj(gca,'Tag','Box');
%     for j=1:length(h)
%         patch(get(h(j),'XData'),get(h(j),'YData'),'y','FaceAlpha',.5);
%     end
        




% 
%     dataObs = test;
%     dataSim = arima2;
%     [nnR,nnENS,nnD,nnLeg,nnRMSE,nnrRMSE,nnMAE,nnrMAE,nnPI] =asseMetric(dataObs,dataSim);
%     arima4_Errors = nnPI;    
    
    
    
%plot of arima models
%     Arimalmodels = reData(:,[1 2 3 11 12 13 14 15 16 17]);
%     figure
%     plot(Arimalmodels);
%     title('Test vs Arima Model Forecast')
%     xlabel('days')
%     ylabel('\Delta [All Ord Price]')
%     var = {{'Test','arima1','arima2','arima3','arima4','arima5','arima6','arima7','arima8','arima9','arima10'},'Location','southeast'}
%     legend(var{:})
%     
%     Arimalmodels = reData(:,[1 2 3 11]);
%     figure
%     plot(Arimalmodels);
%     title('Test vs Arima Model Forecast')
%     xlabel('days')
%     ylabel('\Delta [All Ord Price]')
%     var = {{'Test','arima1','arima2','arima3','arima4'},'Location','southeast'}
%     legend(var{:})
%     
%     Arimalmodels = reData(:,[1 12 13 14]);
%     figure
%     plot(Arimalmodels);
%     title('Test vs Arima Model Forecast')
%     xlabel('days')
%     ylabel('\Delta [All Ord Price]')
%     var = {{'Test','arima5','arima6','arima7'},'Location','southeast'}
%     legend(var{:})
%         
%     Arimalmodels = reData(:,[1 15 16 17]);
%     figure
%     plot(Arimalmodels);
%     title('Test vs Arima Model Forecast')
%     xlabel('days')
%     ylabel('\Delta [All Ord Price]')
%     var = {{'Test','arima8','arima9','arima10'},'Location','southwest'}
%     legend(var{:})   
   
    subplot(2,2,1)
    dataErr = abs(test - en5ANFIS);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hold on;
    hist(dataErr,[hmin:10:hmax]);
    title({'ANFIS-Ensemble 5','error frequency'})
    ylabel('Frequency of Error');
    xlabel('Errors (brackets ± 10)');
    hold off;    
    subplot(2,2,2)
    dataErr = abs(test - en10ANFIS);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hold on;
    hist(dataErr,[hmin:10:hmax]);
    title({'ANFIS-Ensemble 10','error frequency'})
    ylabel('Frequency of Error');
    xlabel('Errors (brackets ± 10)');
    hold off; 
    subplot(2,2,3)
    dataErr = abs(test - en50ANFIS);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hold on;
    hist(dataErr,[hmin:10:hmax]);
    title({'ANFIS-Ensemble 50','error frequency'})
    ylabel('Frequency of Error');
    xlabel('Errors (brackets ± 10)');
    hold off; 
    subplot(2,2,4)
    dataErr = abs(test - en100ANFIS);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hold on;
    hist(dataErr,[hmin:10:hmax]);
    title({'ANFIS-Ensemble 100','error frequency'})
    ylabel('Frequency of Error');
    xlabel('Errors (brackets ± 100)');
    hold off;    
    
    

%     %subplot(1,1,1)
%     violin([test arima1 arima2 t1mamdami t1Sugerno allSugerno nt1Sugerno en50ANFIS en100ANFIS SOMANFIS],...
%         'facecolor',[[1 0 0];[0 0 1];[0 0 1];[0 0 1];[0 0 1];[0 0 1];[0 0 1];[0 0 1];[0 0 1];[0 0 1]],...
%         'medc','','mc','k')
    
%         %Kernel Desity function
%         'edgecolor','b',...
%         'bw',0.3,...
%         'mc','k',...
%         'medc','r--')
%         ylabel('\Delta [All Ord Price]','FontSize',14)  

    
    %'xlabel',{'Test','arima1','arima2','t1mamdami','t1Sugerno','allSugerno'},...
    %'xlabel',{'a','b','c','d','a','b','c','d','c','d'},...
    % xlabel:    xlabel. Set either [] or in the form {'txt1','txt2','txt3',...}
    figure
    plot(models);
    title('Test vs Model Forecast')
    xlabel('days')
    ylabel('\Delta [All Ord Price]')
    var = {{'Test','arima1','arima2','t1mamdami','t1Sugerno','allSugerno','nt1Sugerno','en50ANFIS','en100ANFIS','SOMANFIS'},'Location','southwest'}
    legend(var{:})
    %legend({'Test','arima1','arima2','t1mamdami','t1Sugerno','allSugerno','nt1Sugerno','en50ANFIS','en100ANFIS','SOMANFIS'},'Location','northwest','NumColumns',2)
 
    
%---------------------------------------------------------------   
elseif ModelRun == 9   % plot inital time series
%--------------------------------------------------------------    

% %Ploting actual stock Indexes
plot(A);
count1 = timeseries(A(:,1),1:size(A,1),'name', 'AORD');
count1.TimeInfo.Units = 'Daily Count';
% count1.TimeInfo.Units = 'days';
% count1.TimeInfo.StartDate = '01-Jan-2008';     % Set start date.
% count1.TimeInfo.Format = 'dd mmm, yy';       % Set format for display on x-axis.
% count1.Time = count1.Time - count1.Time(1);        % Express time relative to the start date.

plot(count1,'k','LineWidth',0.01)
grid on
hold on
count2 = timeseries(A(:,2),1:size(A,1),'name', 'DJI');
plot(count2,':r')
count3 = timeseries(A(:,3),1:size(A,1),'name', 'FTSE');
plot(count3,':g')
count4 = timeseries(A(:,4),1:size(A,1),'name', 'HSI');
plot(count4,':c')
count5 = timeseries(A(:,5),1:size(A,1),'name', 'N225');
plot(count5,':m')
title('Time Series: Daily Five Stock Indexes Price')
xlabel('Time (Days)')
ylabel('Price (Actual)')
legend('AORD','DJI','FTSE','HSI','N225','Location','northwest')   




% % Ploting normalise stock Indexes
% plot(A_Final);
% count1 = timeseries(A_Final(:,2),1:size(A_Final,1),'name', 'AORD');
% count1.TimeInfo.Units = 'Daily Count';
% % count1.TimeInfo.Units = 'days';
% % count1.TimeInfo.StartDate = '01-Jan-2008';     % Set start date.
% % count1.TimeInfo.Format = 'dd mmm, yy';       % Set format for display on x-axis.
% % count1.Time = count1.Time - count1.Time(1);        % Express time relative to the start date.
% 
% plot(count1,'k','LineWidth',0.01)
% grid on
% hold on
% count2 = timeseries(A_Final(:,3),1:size(A_Final,1),'name', 'DJI');
% plot(count2,':r')
% count3 = timeseries(A_Final(:,4),1:size(A_Final,1),'name', 'FTSE');
% plot(count3,':g')
% count4 = timeseries(A_Final(:,5),1:size(A_Final,1),'name', 'HSI');
% plot(count4,':c')
% count5 = timeseries(A_Final(:,6),1:size(A_Final,1),'name', 'N225');
% plot(count5,':m')
% title('Time Series: Normailised Five Stock Indexes')
% xlabel('Time (Days)')
% ylabel('Price (Normalised)')
% legend('AORD','DJI','FTSE','HSI','N225','Location','southeast')   
%     
end
