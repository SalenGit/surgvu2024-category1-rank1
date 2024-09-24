import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
import random
from tqdm import tqdm
import json
import sys
import os
import glob


####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
####
execute_in_docker = False


class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print('File found: ' + str(path))
        if ((str(path)[-3:])) == 'mp4':
            if not path.is_file():
                raise IOError(
                    f"Could not load {fname} using {self.__class__.__qualname__}."
                )
                #cap = cv2.VideoCapture(str(fname))
            #return [{"video": cap, "path": fname}]
            return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )

class Surgtoolloc_det(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-tools.json") if execute_in_docker else Path(
                            "./output/surgical-tools.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights
        ###                                                                                                     ###
        
        
        self.tool_list = ["needle_driver",
                          "monopolar_curved_scissor",
                          "force_bipolar",
                          "clip_applier",
                          "tip_up_fenestrated_grasper",
                          "cadiere_forceps",
                          "bipolar_forceps",
                          "vessel_sealer",
                          "suction_irrigator",
                          "bipolar_dissector",
                          "prograsp_forceps",
                          "stapler",
                          "permanent_cautery_hook_spatula",
                          "grasping_retractor"]

        self.num_to_lab = {
                        0: 'bipolar_forceps', 
                        1: 'needle_driver', 
                        2: 'cadiere_forceps', 
                        3: 'monopolar_curved_scissor', 
                        4: 'vessel_sealer', 
                        5: 'force_bipolar', 
                        6: 'prograsp_forceps', 
                        7: 'permanent_cautery_hook_spatula', 
                        8: 'stapler', 
                        9: 'grasping_retractor', 
                        10: 'clip_applier', 
                        11: 'tip_up_fenestrated_grasper', 
                        12: 'suction_irrigator', 
                        }

    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py

        # Write resulting candidates to result.json for this case
        return dict(type="Multiple 2D bounding boxes", boxes=scored_candidates, version={"major": 1, "minor": 0})

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def generate_bbox(self, frame_id, width, height, all_dets, all_index):
        # bbox coordinates are the four corners of a box: [x, y, 0.5]
        # Starting with top left as first corner, then following the clockwise sequence
        # origin is defined as the top left corner of the video frame

        predictions = []
        
        if frame_id in all_index:
            det_file = all_dets[all_index.index(frame_id)]
            
            with open(det_file, 'r') as f:
                lines = f.readlines()
                f.close()
                
            
            for line in lines:
        
                cls, x_c, y_c, w, h, prob = map(float, line.split())
                
                name = f'slice_nr_{frame_id}_' + self.num_to_lab[int(cls)]
                bbox = [[(x_c-w/2)*width, (y_c-h/2)*height, 0.5],
                        [(x_c+w/2)*width, (y_c-h/2)*height, 0.5],
                        [(x_c+w/2)*width, (y_c+h/2)*height, 0.5],
                        [(x_c-w/2)*width, (y_c+h/2)*height, 0.5]]
                
                prediction = {"corners": bbox, "name": name, "probability": prob}
                predictions.append(prediction)
            
        return predictions

    def predict(self, fname) -> DataFrame:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        
        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###
        os.system('rm -r runs')
        if execute_in_docker:
            os.system(f'{sys.argv[-1]} detect.py --source {fname} --half --img 1280 --iou-thre 0.75 --conf-thres 0.001 --max-det 20 --augment --weights /opt/algorithm/stage_0_weights/x_1280_220k.pt --name "video_test" --save-txt --save-conf --nosave')
        else:
            os.system(f'{sys.argv[-1]} detect.py --source {fname} --half --img 1280 --iou-thre 0.75 --conf-thres 0.001 --max-det 20  --augment --weights stage_0_weights/x_1280_220k.pt --name "video_test" --save-txt --save-conf')
        all_dets = glob.glob('runs/detect/video_test/labels/*.txt')
        all_index = [int(i.split('/')[-1].split('.')[0].split('_')[-1])-1 for i in all_dets]
        
        all_frames_predicted_outputs = []
        print('parsing yolo det result...')
        for fid in tqdm(range(num_frames)):
            tool_detections = self.generate_bbox(fid, width, height, all_dets, all_index)
            all_frames_predicted_outputs += tool_detections

        os.system('rm -r runs')
        
        return all_frames_predicted_outputs


if __name__ == "__main__":
    Surgtoolloc_det().process()
