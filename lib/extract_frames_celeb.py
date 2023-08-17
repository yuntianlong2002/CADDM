#!/usr/bin/env python3
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import dlib
import json
import argparse
from imutils import face_utils


VIDEO_PATH = "./data/Celeb-DF"  # Change to your Celeb-DF root directory
SAVE_IMGS_PATH = "./test_images_df"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
TESTING_VIDEO_LIST = os.path.join(VIDEO_PATH, "List_of_testing_videos.txt")
IMG_META_DICT = dict()
NUM_FRAMES = 1


def parse_video_paths_and_labels():
    # Read the list of testing videos and their labels
    with open(TESTING_VIDEO_LIST, 'r') as f:
        lines = f.readlines()

    video_paths = []
    labels = []
    for line in lines:
        label, rel_path = line.strip().split()
        video_path = os.path.join(VIDEO_PATH, rel_path)
        video_paths.append(video_path)
        labels.append(int(label))

    return video_paths, labels


def parse_source_save_path(save_path):
    source_save_path = save_path
    return source_save_path


def preprocess_video(video_path, save_path, label, face_detector, face_predictor):
    # save the video meta info here
    video_dict = dict()
    # get the path of corresponding source imgs
    source_save_path = parse_source_save_path(save_path)
    # prepare the save path
    os.makedirs(save_path, exist_ok=True)
    # read the video and prepare the sampled index
    cap_video = cv2.VideoCapture(video_path)
    frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count_video - 1, NUM_FRAMES, endpoint=True, dtype=np.int)
    # process each frame
    for cnt_frame in range(frame_count_video):
        ret, frame = cap_video.read()
        height, width = frame.shape[:-1]
        if not ret:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(video_path)))
            continue
        if cnt_frame not in frame_idxs:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector(frame, 1)
        if len(faces) == 0:
            tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(video_path)))
            continue
        landmarks = list()  # save the landmark
        size_list = list()  # save the size of the detected face
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        # save the landmark with the biggest face
        landmarks = np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]
        # save the meta info of the video
        video_dict['landmark'] = landmarks.tolist()
        video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}.png"
        video_dict['label'] = label
        IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = video_dict
        # save one frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)
    cap_video.release()
    return


def main():
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    video_paths, labels = parse_video_paths_and_labels()
    n_sample = len(video_paths)
    for i in tqdm(range(n_sample)):
        save_path_per_video = video_paths[i].replace(
            VIDEO_PATH, SAVE_IMGS_PATH
        ).replace('.mp4', '').replace("/videos", "/frames")
        preprocess_video(
            video_paths[i], save_path_per_video, labels[i],
            face_detector, face_predictor
        )
    with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
        json.dump(IMG_META_DICT, f)


if __name__ == '__main__':
    main()