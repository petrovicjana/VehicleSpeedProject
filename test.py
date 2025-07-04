from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("data/videos/speed_estimation.mp4", save=True, show=True)


# we will have to resize frames, run on gpu in order for yolo to perform good 
