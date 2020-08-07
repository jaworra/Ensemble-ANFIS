function [cellPACF] = funPACF(indexRan);
%PACF 
for k = 1:length(indexRan)
    index= readtable(indexRan{k});
    dates= table2cell(index(:,1));
    %dn{k} = datenum(dates, 'yyyy-mm-dd');
    dn = datenum(dates, 'yyyy-mm');
    closeVal = table2array(index(:,5));
    
    %Dealing with markets that dont open (NA values)
    %Change Null to 0
    if isa(closeVal,'double') == 0 % Run if array has NAN
%        %NA to 0 value
        closeVal = cellfun(@str2double, closeVal);
        closeVal(isnan(closeVal))=''; 
%        %closeVal(isnan(closeVal))=""; 
%        %closeVal(isnan(closeVal))="0"; 
    end   
   
    %PACF Figures
    cellPACF{k} = parcorr(closeVal);
    figure(1)
    subplot(2,3,k)
    parcorr(closeVal,10)
    name = indexRan{k}(1:end-4);
    title(strcat('PACF of  ',{' '},name));
    xlabel('lag (in days)')
    ylabel('')
    %ylabel('PACF value')
        
    figure(2)
    %plot(dn,closeVal);
    plot(closeVal);
    xlabel('Days')
    ylabel('Index Value')
    legendInfo{k} = [name]; 
    hold on
    if k == length(indexRan) % last value add Legend and Title
        str = strjoin(legendInfo,', ')
        title(strcat('Index Time Series of ',{' '},str));
        legend(legendInfo)
        hold off
    end
   
end
