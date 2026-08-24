from ultralytics import YOLO

model = YOLO("runs/detect/123123/weights/best.pt")

if __name__ == '__main__':
    results = model.val(
        data="LLVIP.yaml",  
        device='0,1',  
        workers=8,  
        batch=32,  
    )
    print(f"mAP75: {results.box.map75}")