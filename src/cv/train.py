from roboflow import Roboflow
from ultralytics import YOLO

def train_model():
    rf = Roboflow(api_key="qXLxOkZBh12NPk9ciVsq")

    project = rf.workspace("mohabs-workspace").project("bender-test-cv")
    dataset = project.version(1).download("yolov8")

    model = YOLO('yolov8n.pt')

    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        augment=True,
        project='models/yolo',
        name='run_1'
    )

if __name__ == "__main__":
    train_model()