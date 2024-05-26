import os
import gradio as gr
import cv2
import subprocess

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro

sample_video = os.path.join(os.path.dirname(__file__), "samples/sample-squats.mp4")

# Initialize face mesh solution
POSE = get_mediapipe_pose()


def process_video(video_path, mode="Beginner"):
    output_video_file = f"output_recorded.mp4"

    if mode == 'Beginner':
        thresholds = get_thresholds_beginner()
    elif mode == 'Pro':
        thresholds = get_thresholds_pro()

    upload_process_frame = ProcessFrame(thresholds=thresholds)

    vf = cv2.VideoCapture(video_path)
    fps = int(vf.get(cv2.CAP_PROP_FPS))
    width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

    count = 0
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break

        # convert frame from BGR to RGB before processing it.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame, _ = upload_process_frame.process(frame, POSE)

        video_output.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

        if not count % 12:
            yield out_frame, None

        count += 1

    vf.release()
    video_output.release()

    yield None, output_video_file


input_video = gr.Video(label="Input Video")

# Correct handling for webcam input
webcam_video = gr.Video(label="Webcam Video")

output_frames_up = gr.Image(label="Output Frames")
output_video_file_up = gr.Video(label="Output Video")

output_frames_cam = gr.Image(label="Output Frames")
output_video_file_cam = gr.Video(label="Output Video")

upload_interface = gr.Interface(
    fn=process_video,
    inputs=[input_video, gr.Radio(choices=["Beginner", "Pro"], label="Select Mode")],
    outputs=[output_frames_up, output_video_file_up],
    title="SOAR Squats Analysis",
    allow_flagging="never",
    examples=[[sample_video]]
)

webcam_interface = gr.Interface(
    fn=process_video,
    inputs=[webcam_video, gr.Radio(choices=["Beginner", "Pro"], label="Select Mode")],
    outputs=[output_frames_cam, output_video_file_cam],
    title="SOAR Squats Analysis",
    allow_flagging="never"
)

app = gr.TabbedInterface([upload_interface, webcam_interface],
                         tab_names=["⬆️ Upload Video", "📷️ Live Stream"])

app.queue().launch()
