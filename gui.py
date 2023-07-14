import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mmcv
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

class WebcamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Webcam App")

        self.video = cv2.VideoCapture(0)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_capture = tk.Button(window, text="Capture", command=self.capture_frame)
        self.btn_capture.pack(pady=10)

        device = 'cpu'
        config = 'pcb_config.py'
        checkpoint = 'pcb_checkpoint.pth'

        # build the model from a config file and a checkpoint file
        device = torch.device(device)
        self.model = init_detector(config, checkpoint, device=device)

        # init visualizer
        self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_detector
        self.visualizer.dataset_meta = self.model.dataset_meta

        self.update()

    def update(self):
        score_thr = 0.5

        ret_val, frame = self.video.read()
        if ret_val:
            result = inference_detector(self.model, frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            # frame_rgb = mmcv.imconvert(frame_rgb, 'rgb', 'rgb')
            self.visualizer.add_datasample(
                name='result',
                image=frame_rgb,
                data_sample=result,
                draw_gt=False,
                pred_score_thr=score_thr,
                show=False)

            frame_rgb = self.visualizer.get_image()
            # frame_rgb = mmcv.imconvert(frame_rgb, 'rgb', 'rgb')
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update)

    def capture_frame(self):
        ret_val, frame = self.video.read()
        if ret_val:
            cv2.imwrite("captured_frame.jpg", frame)
            print("Frame captured!")

if __name__ == '__main__':
    window = tk.Tk()
    app = WebcamApp(window)
    window.mainloop()
