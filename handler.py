import runpod

def handler(job):
    return {"status": "success", "message": "Hello from RunPod!"}

runpod.serverless.start({"handler": handler})
