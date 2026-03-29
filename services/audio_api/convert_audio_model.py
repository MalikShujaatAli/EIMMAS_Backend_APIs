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
        
        # ✅ THE ULTIMATE FIX: Force a STATIC batch size of 1.
        # This removes the "None" dimension and gives TFLite the exact 
        # memory blueprint it needs to optimize the BiLSTM layer natively.
        run_model = tf.function(lambda x: model(x))
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec(shape=(1, 174, 40), dtype=tf.float32)
        )

        print("🔄 Converting to TFLite (Static Shapes & Optimized for CPU)...")
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Optimization settings for speed
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Force Native TFLite Built-ins ONLY
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter._experimental_lower_tensor_list_ops = True
        
        tflite_model = converter.convert()
        
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"\n✅ SUCCESS: '{tflite_path}' generated.")
        print("🚀 Flex Delegate bug annihilated! Your Audio API will now load at lightning speed.")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")

if __name__ == "__main__":
    convert()