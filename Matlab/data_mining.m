function dataset = data_mining(path, first_year, last_year)

% This function builds a table with energy demand, weather variables and calendar variables
%
% INPUT:
% path:         path of the original dataset gefcom.csv
% first_year:   first year of the output dataset
% last_year:    last year of the output dataset
%
% OUTPUT:
% dataset: table containing energy demand, weather variables and calendar variables

if nargin < 2
    first_year = 2008;
    last_year = 2016;
elseif nargin == 2
    last_year = inf;
end

%% Load data
raw_data = readtable(path);
df = raw_data;

%% Filter data

% consider the whole New England area
df.zone = char((df.zone(:)));
df = df(strcmp(df.zone, "TOTAL"), :);

% remove the 29th of February
day_number = datestr(datenum(df.date), 'dd');

isFeb = strcmp(df.month, "Feb");
is29 = strcmp(day_number, "29");

df = df(not(isFeb.*is29), :);

% consider only the years between 2008 and 2016
df = df((df.year>=first_year) & (df.year<=last_year), :);
    
%% Aggregate data
% Create output dataset, considering daily consumption and average weather conditions for every day

[group, id] = findgroups(df.date);

measures = splitapply(@(q,r,s) [sum(q), mean(r), mean(s)], ...
   df.demand, df.dewpnt, df.drybulb, group);
dates = splitapply(@(t) t(1), datenum(df.date), group);
calendar = splitapply(@(x,y,z) [x(1), y(1),z(1)],...
    df.month, df.day_of_week, df.holiday, group);
years = splitapply(@(t) t(1), datenum(df.year), group);

output0 = table(dates);
output1 = table(years);
output3 = array2table(measures, 'VariableNames', ['demand',"dewpnt",'drybulb']);
output2 = array2table(calendar, 'VariableNames', ['month',"day_of_week",'holiday']);

dataset = [output0, output1, output2, output3];
dataset.month = char((dataset.month(:)));
dataset.day_of_week = char((dataset.day_of_week(:)));
dataset.holiday = char((dataset.holiday(:)));
dataset.holiday = strcmp(dataset.holiday, "TRUE");

end