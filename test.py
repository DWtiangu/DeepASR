from config import model_save_dir, cache_dir, stft_fea, mel_fea,tf_mel_fea, label_type, batch_size
from Scripts.Models.DataParsers.AcousticParser import AcousticDataParser, AcousticLabelParser
from Scripts.Data.Thchs30Data import Thchs30Data
from Scripts.utils.others import test_dataObjs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
DataOBJ,example_fp = (Thchs30Data,"A11_0.wav")
dataObj = DataOBJ(filepath = example_fp)

tf_mel_dataparser = AcousticDataParser(
    feature=tf_mel_fea, cache_dir=cache_dir, open_cache = False)

data = tf_mel_dataparser(dataObj)
print(data,data.shape)


import pandas as pd

