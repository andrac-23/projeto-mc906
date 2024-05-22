from ultralytics import YOLO

# Load a model
model = YOLO(r"localizacao_best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model([r"oi.jpeg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print(result.boxes.xyxy)
    result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk
