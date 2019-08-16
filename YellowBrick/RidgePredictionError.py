from sklearn.linear_model import Ridge
from yellowbrick.regressor import PredictionError
import bikeshare

visualizer = PredictionError(Ridge(alpha=3.181))
visualizer.fit(bikeshare.X_train, bikeshare.y_train)
visualizer.score(bikeshare.X_test, bikeshare.y_test)
visualizer.poof()
