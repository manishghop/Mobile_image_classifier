import tensorflow as tf

model_path = "mobile_fruit_classification.pb"
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# convert the model 
tf_lite_model = converter.convert()
# save the converted model 
open('mobile_classifier.tflite', 'wb').write(tf_lite_model)