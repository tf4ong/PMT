#imported from deeplabcut 
#DeepLabCut2.0 Toolbox (deeplabcut.org)
import cv2
import datetime
import numpy as np
import os
import subprocess
import warnings
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

class VideoReader:
    def __init__(self, video_path):
        if not os.path.isfile(video_path):
            raise ValueError(f'Video path "{video_path}" does not point to a file.')
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise IOError("Video could not be opened; it may be corrupted.")
        self.parse_metadata()
        self._bbox = 0, 1, 0, 1
        self._n_frames_robust = None

    def __repr__(self):
        string = "Video (duration={:0.2f}, fps={}, dimensions={}x{})"
        return string.format(self.calc_duration(), self.fps, *self.dimensions)

    def __len__(self):
        return self._n_frames

    def check_integrity(self):
        dest = os.path.join(self.directory, f"{self.name}.log")
        command = f"ffmpeg -v error -i {self.video_path} -f null - 2>{dest}"
        subprocess.call(command, shell=True)
        if os.path.getsize(dest) != 0:
            warnings.warn(f'Video contains errors. See "{dest}" for a detailed report.')

    @property
    def name(self):
        return os.path.splitext(os.path.split(self.video_path)[1])[0]

    @property
    def format(self):
        return os.path.splitext(self.video_path)[1]

    @property
    def directory(self):
        return os.path.dirname(self.video_path)

    @property
    def metadata(self):
        return dict(
            n_frames=len(self), fps=self.fps, width=self.width, height=self.height
        )

    def get_n_frames(self, robust=False):
        if not robust:
            return self._n_frames
        elif not self._n_frames_robust:
            command = (
                f'ffprobe -i "{self.video_path}" -v error -count_frames '
                f"-select_streams v:0 -show_entries stream=nb_read_frames "
                f"-of default=nokey=1:noprint_wrappers=1"
            )
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
            self._n_frames_robust = int(output)
        return self._n_frames_robust

    def calc_duration(self, robust=False):
        if robust:
            command = (
                f'ffprobe -i "{self.video_path}" -show_entries '
                f'format=duration -v quiet -of csv="p=0"'
            )
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
            return float(output)
        return len(self) / self.fps

    def set_to_frame(self, ind):
        if ind < 0:
            raise ValueError("Index must be a positive integer.")
        last_frame = len(self) - 1
        if ind > last_frame:
            warnings.warn(
                "Index exceeds the total number of frames. "
                "Setting to last frame instead."
            )
            ind = last_frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, ind)

    def reset(self):
        self.set_to_frame(0)

    def read_frame(self, shrink=1, crop=False):
        success, frame = self.video.read()
        if not success:
            return
        frame = frame[..., ::-1]  # return RGB rather than BGR!
        if crop:
            x1, x2, y1, y2 = self.get_bbox(relative=False)
            frame = frame[y1:y2, x1:x2]
        if shrink > 1:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (w // shrink, h // shrink),
                fx=0,
                fy=0,
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def get_bbox(self, relative=False):
        x1, x2, y1, y2 = self._bbox
        if not relative:
            x1 = int(self._width * x1)
            x2 = int(self._width * x2)
            y1 = int(self._height * y1)
            y2 = int(self._height * y2)
        return x1, x2, y1, y2

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        if not fps > 0:
            raise ValueError("Frame rate should be positive.")
        self._fps = fps

    @property
    def width(self):
        x1, x2, _, _ = self.get_bbox(relative=False)
        return x2 - x1

    @property
    def height(self):
        _, _, y1, y2 = self.get_bbox(relative=False)
        return y2 - y1

    @property
    def dimensions(self):
        return self.width, self.height

    def parse_metadata(self):
        self._n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._n_frames >= 1e9:
            warnings.warn(
                "The video has more than 10^9 frames, we recommend chopping it up."
            )
        self._width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = round(self.video.get(cv2.CAP_PROP_FPS), 2)

    def close(self):
        self.video.release()

def KmeansbasedFrameselectioncv2(
    cap,
    numframes2pick,
    start,
    stop,
    crop,
    coords,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """ This code downsamples the video to a width of resizewidth.
    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.
    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.
    Note: this method can return fewer images than numframes2pick.
    Attention: the flow of commands was not optimized for readability, but rather speed. This is why it might appear tedious and repetetive."""
    nframes = len(cap)
    nx, ny = cap.dimensions
    ratio = resizewidth * 1.0 / nx
    if ratio > 1:
        raise Exception("Choice of resizewidth actually upsamples!")

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = nframes // 2

    allocated = False
    if len(Index) >= numframes2pick:
        if (
            np.mean(np.diff(Index)) > 1
        ):  # then non-consecutive indices are present, thus cap.set is required (which slows everything down!)
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame()
                    if frame is not None:
                        if crop:
                            frame = frame[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ]

                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1] * 3)
                            )
                            allocated = True
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame()
                    if frame is not None:
                        if crop:
                            frame = frame[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ]
                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1])
                            )
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, 2)
        else:
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame()
                    if frame is not None:
                        if crop:
                            frame = frame[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ]

                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1] * 3)
                            )
                            allocated = True
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame()
                    if frame is not None:
                        if crop:
                            frame = frame[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ]
                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1])
                            )
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, 2)

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )
        # cap.release() >> still used in frame_extraction!
        return list(np.array(frames2pick))
    else:
        return list(Index)