import pandas as pd
from yellowbrick.features import Rank2D

data = pd.read_csv('../CSV/bikeshare.csv')
X = data[[
    "season", "month", "hour", "holiday", "weekday", "workingday",
    "weather", "temp", "feelslike", "humidity", "windspeed"
]]
y = data["riders"]

visualizer = Rank2D(algorithm="pearson")
visualizer.fit_transform(X)
visualizer.poof()

'''
This figure shows us the Pearson correlation between pairs of features such that each cell in the grid represents 
two features identified in order on the x and y axes and whose color displays the magnitude of the correlation. 
A Pearson correlation of 1.0 means that there is a strong positive, linear relationship between the pairs of 
variables and a value of -1.0 indicates a strong negative, linear relationship (a value of zero indicates no 
relationship). Therefore we are looking for dark red and dark blue boxes to identify further.

In this chart, we see that the features temp and feelslike have a strong correlation and also that the feature 
season has a strong correlation with the feature month. This seems to make sense; the apparent temperature we feel 
outside depends on the actual temperature and other airquality factors, and the season of the year is described by 
the month!
'''