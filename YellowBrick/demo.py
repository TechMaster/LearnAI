import bikeshare
import matplotlib.pyplot as plt
from yellowbrick.features import JointPlotVisualizer


visualizer = JointPlotVisualizer(feature='temp', target='feelslike')
visualizer.fit(bikeshare.X['temp'], bikeshare.X['feelslike'])

visualizer.poof()
plt.show()
