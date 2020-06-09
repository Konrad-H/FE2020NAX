function dataset = data_mining(path,first_year,last_year)

if nargin<2
    first_year=2008
    last_year =2016
elseif nargin==2
    last_year=inf
end
%% LOAD DATA
raw_data = readtable(path);

%%
df = raw_data;
df.zone = char((df.zone (:)));
%% FILTER DATA


% ONLY TOTAL data is useful

df= df( strcmp(df.zone , "TOTAL"),:);
% Takeout february
day_number = datestr(datenum(df.date),'dd');

isFeb = strcmp(df.month,"Feb");
is29 = strcmp( day_number, "29");

df = df( not (isFeb.*is29),:);

% Takeout extra years

df =  df(  (df.year>=first_year)&(df.year<=last_year),:);
    
%% AGGREGATE DATA

[group, id] = findgroups(df.date);

measures = splitapply(@(q,r,s) [sum(q), mean(r), mean(s)], ...
   df.demand, df.dewpnt, df.drybulb, group);
dates = splitapply(@(t) t(1), datenum(df.date), group);
calendar = splitapply(@(x,y,z) [x(1), y(1),z(1)],...
    df.month,	df.day_of_week,	df.holiday, group);
years = splitapply(@(t) t(1), datenum(df.year), group);

output0=table(dates);
output1=table(years);
output3=array2table(measures,'VariableNames',['demand',"dewpnt",'drybulb']);
output2=array2table(calendar,'VariableNames',['month',"day_of_week",'holiday']);

dataset = [output0, output1, output2, output3];
dataset.month = char((dataset.month (:)));
dataset.day_of_week = char((dataset.day_of_week (:)));
dataset.holiday = char((dataset.holiday (:)));
dataset.holiday = strcmp(dataset.holiday,"TRUE");
end