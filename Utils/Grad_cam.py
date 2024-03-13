import os
import tensorflow as tf
# Other
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Main
def _compute_heatmap(model,
                     img_array,
                     conv_layer_name,
                     pred_index):
    """
    Helper function to compute the heatmap for a given convolutional layer.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_layer_output = conv_layer_output[0]
    heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap

def make_gradcam_heatmap(img_array,
                         model,
                         last_conv_layer_name,
                         second_last_conv_layer_name=None,
                         pred_index=None,
                         sensitivity_map=1.0):
    """
    Function to compute the Grad-CAM heatmap for a specific class, given an input image.
    """
    if pred_index is None:
        preds = model.predict(img_array)
        pred_index = tf.argmax(preds[0])

    # Compute heatmap for the last convolutional layer
    heatmap = _compute_heatmap(model, img_array, last_conv_layer_name, pred_index)
    heatmap = heatmap ** sensitivity_map

    if second_last_conv_layer_name is not None:
        # Compute heatmap for the second last convolutional layer
        heatmap_second = _compute_heatmap(model, img_array, second_last_conv_layer_name, pred_index)
        heatmap_second = heatmap_second ** sensitivity_map
        
        # Average the two heatmaps
        heatmap = (heatmap + heatmap_second) / 2.0
    
    return heatmap