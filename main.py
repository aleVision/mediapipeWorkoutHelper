
import numpy as np
import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import text
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from utils import draw_landmarks_on_image

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class WorkoutHelper():
  def __init__(self) -> None:
    self.result = None
    self.mp_drawing = solutions.drawing_utils
    self.mp_pose = solutions.pose

  # Create a pose landmarker instance with the live stream mode:
  def print_result(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
      #print('pose landmarker result: {}'.format(result.pose_landmarks))
      self.result = result  

  def main(self):
      base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
      options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True, 
                                            running_mode=VisionRunningMode.LIVE_STREAM, result_callback=self.print_result)    

      with PoseLandmarker.create_from_options(options) as landmarker:
        video = cv2.VideoCapture(0)

        while(True):
          _, frame = video.read()
          frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)        

          if not _:
            break
          
          # Convert the frame received from OpenCV to a MediaPipe’s Image object.
          mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
          landmarker.detect_async(mp_image, mp.Timestamp.from_seconds(time.time()).value)
          
          if self.result is not None:
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), self.result)
            cv2.imshow('Pose view', annotated_image)
          
          else: cv2.imshow('Pose view', frame)

          if cv2.waitKey(25) & 0xFF == ord('q'):                
            break

        video.release()
        cv2.destroyAllWindows()      

if __name__ == "__main__":
  coach = WorkoutHelper()
  coach.main()
