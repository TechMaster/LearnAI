import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.features import JointPlotVisualizer

data = pd.read_csv('../CSV/bikeshare.csv')
X = data[[
    "season", "month", "hour", "holiday", "weekday", "workingday",
    "weather", "temp", "feelslike", "humidity", "windspeed"
]]
y = data["riders"]

visualizer = JointPlotVisualizer(feature='temp', target='feelslike')
visualizer.fit(X['temp'], X['feelslike'])

visualizer.poof()
plt.show()
