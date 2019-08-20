TensorFlow 2.0 yêu cầu numpy 1.16.4
pip install numpy==1.16.4

Nếu cài bản numpy 1.17 sẽ báo warning "FutureWarning: Deprecated numpy API calls in tf.python.framework.dtypes"


Khi nâng lên TensorFlow 2.0
AttributeError: module 'tensorflow' has no attribute 'placeholder'

# Cách xử lý

Thêm đoạn code này vào code viết cho version 1
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```