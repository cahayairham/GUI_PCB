import cv2
import mmcv
import torch

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

device = 'cpu'
config = 'pcb_config.py'
checkpoint = 'pcb_checkpoint.pth'
camera_id = 0
score_thr = 0.5


# build the model from a config file and a checkpoint file
device = torch.device(device)
model = init_detector(config, checkpoint, device=device)

# init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

camera = cv2.VideoCapture(camera_id)

print('Press "Esc", "q" or "Q" to exit.')
while True:
    ret_val, img = camera.read()
    result = inference_detector(model, img)

    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=score_thr,
        show=False)

    img = visualizer.get_image()
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    cv2.imshow('result', img)

    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord('q') or ch == ord('Q'):
        break

