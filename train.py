from ultralytics import YOLO
'''每个新终端都要初始化一次才能双卡
export PYTHONPATH=/opt/data/private/yolo/ultralytics_v5

创建新会话tmux new -s yolo_train
启动训练
分离会话按下 Ctrl + B,然后按 D
重连会话tmux attach -t yolo_train
鼠标tmux set-option -g mouse on
杀死进程pkill -9 -f python
执行脚本/root/miniconda3/envs/yolo/bin/python /opt/data/private/yolo/ultralytics_v5/train.py
'''
def main():
    model = YOLO("AFCNet.yaml")  # build a new model from scratch
    train_results = model.train(
    #resume=True,  # resume from last training
    data="DroneVehicle.yaml",  # path to dataset YAML  DroneVehicle
    epochs=400,  # number of training epochs
    device='0,1',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers=8,
    batch=32,
    name="123123",  # save to runs/train/baseline
    amp=False,
    patience=50
    )


if __name__ == '__main__':
    main()

