import tensorflow as tf
import os

# Change directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model_path = "audio_best_model.keras"
tflite_path = "audio_model.tflite"

def convert():
    if not os.path.exists(model_path):
        print(f"❌ Error: '{model_path}' not found in {os.getcwd()}.")
        return

    print(f"🔄 Loading {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("🔄 Converting to TFLite (Optimized for CPU Latency)...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimization settings for speed
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # ✅ THE FIX: Allow Select TF Ops for BiLSTM dynamic loops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        tflite_model = converter.convert()
        
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"\n✅ SUCCESS: '{tflite_path}' generated.")
        print("Latency should now be ~10x-20x faster on CPU.")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")

if __name__ == "__main__":
    convert()