from core.frame_extraction import *
from absl.flags import FLAGS
from absl import flags, app
from skimage.util import img_as_ubyte
from skimage import io
import time
import os

#from Deeplabcut 


flags.DEFINE_integer('numframes2pick', 50, 'number of frames to extract for labeling')
flags.DEFINE_string('video_folder','./data','path to videos')
flags.DEFINE_string('output_path','./data','output patht to frames')

def extract_frames(vid_path, n_frames,output_path):
    cap = VideoReader(vid_path)
    frames2pick = KmeansbasedFrameselectioncv2(cap,n_frames, 0, 1,
                                                None,None, step=1,
                                                resizewidth=30,
                                                color=False)
    count = 0
    for index in frames2pick:
        cap.set_to_frame(index)
        frame = cap.read_frame()
        if frame is not None:
            image = img_as_ubyte(frame)
            img_name = (str(output_path)+ f"/{int(time.time())}_img"+ str(count)+ ".png")
            io.imsave(img_name, image)
            count+=1
    cap.close()
    print(f'finished extracting {n_frames} from video:')
    print(f'{vid_path}')

def main(_argv):
    output_path=FLAGS.output_path+'labeled_data'
    print(output_path)
    vids=[FLAGS.video_folder+i+'/raw.avi' for i in os.listdir(FLAGS.video_folder)]
    os.mkdir(output_path)
    for i in vids:
        print(f'Processing video folder {i}')
        extract_frames(i, FLAGS.output_path)

    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
