import runpod
import sys
import os

print("Step 1: Basic imports done")

try:
    print("Step 2: Installing git clone...")
    result = os.system("git clone https://github.com/shiyu-coder/Kronos.git /tmp/Kronos 2>&1")
    print(f"Step 2: Clone result: {result}")
    
    print("Step 3: Adding path...")
    sys.path.append('/tmp/Kronos')
    
    print("Step 4: Importing torch...")
    import torch
    print(f"Step 4: Torch version: {torch.__version__}")
    print(f"Step 4: CUDA available: {torch.cuda.is_available()}")
    
    print("Step 5: Importing yfinance...")
    import yfinance as yf
    print("Step 5: yfinance OK")
    
    print("Step 6: Importing Kronos model...")
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("Step 6: Kronos imported OK")
    
    print("Step 7: Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    print("Step 7: Tokenizer loaded OK")
    
    print("Step 8: Loading model...")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    print("Step 8: Model loaded OK")
    
    print("Step 9: Creating predictor...")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print("Step 9: Predictor created OK")
    
    MODEL_LOADED = True
    print("✅ All steps completed!")

except Exception as e:
    import traceback
    print(f"❌ Failed: {e}")
    print(traceback.format_exc())
    MODEL_LOADED = False
    predictor = None

def handler(job):
    if not MODEL_LOADED:
        return {"status": "error", "message": "Model failed to load", "loaded": False}
    return {"status": "success", "message": "Model loaded!", "loaded": True}

runpod.serverless.start({"handler": handler})
