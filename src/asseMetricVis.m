function asseMetricVis(dataObs,dataSim,nnR,style,titleN)
%Function requiring forcasted, observed values and optional values
%nnR - R^2 value
%style - options for display of figures (1 or 2)
%titleN - title to be included (string)

clear title xlabel ylabel

n = get(gcf,'Number');%set figure number

[m1,n1] = size(dataSim);
[m2,n2] = size(dataObs);

%check dimension of matrix to set limits for x axis
if n1 == 1 && n2 == 1
    ylimMax1 = max(dataSim);
    ylimMax2 = max(dataObs);
    ylimMax = max(ylimMax1,ylimMax2);
    ylimMin1 = min(dataSim);
    ylimMin2 = min(dataObs);
    ylimMin = min(ylimMin1,ylimMin2);

else
    ylimMax1 = max(dataSim)
    ylimMax1 = max(ylimMax1)
    ylimMax2 = max(dataObs)
    ylimMax = max(ylimMax1,ylimMax2);
    ylimMin1 = min(dataSim);
    ylimMin1 = min(ylimMin1);
    ylimMin2 = min(dataObs);
    ylimMin = min(ylimMin1,ylimMin2);
end 
     ylimMin = ylimMin-100
     ylimMax = ylimMax+100

if style == 1  %Tradition visualisation with each figure has separate window
    
    %histogram
    figure(n+1)
    n = n+1;
    dataErr = abs(dataObs - dataSim);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hold on;
    hist(dataErr,[hmin:1:hmax]);
    title({titleN,'Histogram absolute error frequency'});
    ylabel('Frequency of Error');
    xlabel('Errors ');
    hold off;
    
    % Scatterplot
    figure(n+1)
    n = n+1;
    hold on;
    scatter(dataObs,dataSim)
    ylim([ylimMin ylimMax])
    %text(2,2, ['R^2 = ' num2str(nnR)]);
    title({titleN,'Scatterplot prediction againts observed'});
    xlabel('Data Observation - Value');
    ylabel('Data Simulated - Value');
    %add line
    coeffs = polyfit(dataObs, dataSim, 1);
    % Get fitted values
    fittedX = linspace(min(dataObs), max(dataObs), 200);
    fittedY = polyval(coeffs, fittedX);
    plot(fittedX, fittedY, 'r-', 'LineWidth', 3);
    legend(['R^2 = ' num2str(nnR)],'line of best fit' ,'Location', 'Best');
    hold off;
    
    %time series
    figure(n+1)
    n = n+1;
    hold on;
    plot(dataObs);
    plot(dataSim);
    ylim([ylimMin ylimMax]);
    title({titleN,'Timeseries prediction againts observed'});
    ylabel('Value of Index');
    xlabel('Time - (days)');
    legend('observed','forecasted','Location', 'Best');
    hold off;

elseif style == 2 %visualisation in one window for easy comparison
    
    figure(n+1)
    %time series
    subplot(2,1,1)
    hold on;
    plot(dataObs);
    plot(dataSim);
    ylim([ylimMin ylimMax]);
    title({titleN,'Timeseries prediction againts observed'});
    ylabel('Value of Index');
    xlabel('Time - (days)');
    legend('observed','forecasted','Location', 'Best');
    hold off;
    
    %Histogram
    subplot(2,2,3) 
    hold on;
    dataErr = abs(dataObs - dataSim);
    hmin = min(dataErr);
    hmax = max(dataErr);
    hist(dataErr,[hmin:1:hmax]);
    title('Histogram absolute error frequency');
    ylabel('Frequency of Error');
    xlabel('Errors ');
    hold off;
    
    % Scatterplot
    subplot(2,2,4)
    hold on;
    scatter(dataObs,dataSim)
    ylim([ylimMin ylimMax])
    title('Scatterplot prediction againts observed');
    xlabel('Data Observation - Value');
    ylabel('Data Simulated - Value');
    %add line
    coeffs = polyfit(dataObs, dataSim, 1);
    % Get fitted values
    fittedX = linspace(min(dataObs), max(dataObs), 200);
    fittedY = polyval(coeffs, fittedX);
    plot(fittedX, fittedY, 'r-', 'LineWidth', 3);
    legend(['R^2 = ' num2str(nnR)],'line of best fit' ,'Location', 'Best');
    hold off;
    
elseif style == 3 %visualisation with box plots
    
    figure(n+1)
    hold on;
    boxplot(dataSim);
    title('All Ords Index Forecasted Distribution')
    xlabel(titleN);
    ylabel('Price of Stock ($)');
    hold off;
     
elseif style == 4 %line graph for use with box plots - Styl5
    
%     figure(n+1)
%     hold on;
%    % boxplot(dataSim.', 'orientation', 'horizontal');
%     boxplot(dataSim.');
%     title('All Ords Index Forecasted Distribution')
%     ylabel('Price of Stock ($)');
     figure(n+1)
     boxplot(dataSim.','symbol','','PlotStyle','traditional','BoxStyle','outline','Colors',[0.7 0.7 0.7],'Notch','off','MedianStyle','line');
%    boxplot(dataSim.','symbol','','PlotStyle','traditional','BoxStyle','outline','Colors',[0 0 0],'Notch','off','MedianStyle','line');
     title('All Ords Index ensemble Forecasted');

     ylim([ylimMin ylimMax])
     hold on; 
     %axis auto
     plot(dataObs,...
    '-o','MarkerSize',3,...
    'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor','red','Color',[0 0 0]);
     
elseif style == 5 %linestyle with box plots   

    %time series
    %plot(dataObs);
    %plot(dataSim);
    plot(dataObs,...
    '-o','MarkerSize',3,...
    'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor','red','Color',[0 0 0]);
    plot(dataSim,...
    '-o','MarkerSize',3,...
    'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor','green','Color',[0 0 0]);

    ylim([ylimMin ylimMax]);
    ylabel('Value of Index');
    xlabel('Time - (days)');
    legend('observed','forecasted','Location', 'Best');
    hold off
 

end

end