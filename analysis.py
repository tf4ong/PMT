import PyRodentTracks as PRT
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('folder_path', 'None', 'path to video folder to analyze')
flags.DEFINE_string('config_path', 'None','path to config file')
flags.DEFINE_integer('n_moue', 0, 'Number of mouse in video')
flags.DEFINE_string('RFID tags', 'None','RFID tags of all mice in video separated by a comma')


def main(_argv):
    rfid_tags = [int(i) for i in FLAGS.]
    temp = PRT.PRT_analysis(FLAGS.folder_path,FLAGS.config_path,)
    temp.detect_mice()
    temp.load_RFID()
    temp.load_dets()
    temp.RFID_match()
    temp.find_activte_mice()
    temp.compile_travel_trjectories()
    temp.generate_track_video()