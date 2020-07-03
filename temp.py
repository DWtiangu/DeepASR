from Scripts.Models.CNN1d_CTC import CNN1d_CTC_PinYin_Sample_lessDropout
from config import model_save_dir, cache_dir, stft_fea, mel_fea,tf_mel_fea, label_type
import os

import tensorflow as tf
# 指定第?块GPU可用
# print(os.popen('nvidia-smi').read())
# which_GPU = input('which_GPU?')
# os.environ["CUDA_VISIBLE_DEVICES"] = which_GPU


load_weight_path = "saved_models/CNN1d_CTC_PinYin_Sample_lessDropout/MagicData/(gpu_n=1)(feature_name=mel)(label_type=pinyin)/best_val_loss(epoch=70)(loss=7.7)(val_loss=10.5).keras_weight"


base_path = os.path.splitext(load_weight_path)[0]
predict_model_rawinput = base_path+"predict_model_rawinput.h5"
converted_predict_model_tflite_path = base_path + "converted_predict_model_rawinput.tflite"

if not os.path.exists(predict_model_rawinput):
    SAMPLE_RATE = 16000
    FFT_S = tf_mel_fea['kwargs']['fft_s']
    HOP_S = tf_mel_fea['kwargs']['hop_s']
    N_MELS = tf_mel_fea['kwargs']['n_mels']
    n_fft = round(FFT_S*SAMPLE_RATE)
    hop_length = round(HOP_S*SAMPLE_RATE)
    # n_mels = tf_mel_fea['kwargs']['n_mels']

    model_obj = CNN1d_CTC_PinYin_Sample_lessDropout('CNN1d_CTC_PinYin_Sample_lessDropout',feature=mel_fea, data_cache_dir=None,label_type=label_type,save_dir = model_save_dir)
    model_obj.load_weight(load_weight_path)
    print("创建tflite_model的h5文件:",predict_model_rawinput)
    from keras.models import Model
    from keras.layers import Input,Lambda

    raw_audio_input = Input(shape=[None])

    from Scripts.Models.DataParsers.Features.mel import raw_audio2log_mel_spec_op
    def extra_spec(raw_audio_input):
        stfts_ = tf.contrib.signal.stft(
            raw_audio_input,
            frame_length=n_fft,
            frame_step=hop_length,
            fft_length=None)
        spectrogram_ = tf.abs(stfts_)
    #         lower_edge_hertz = 125.0
    #         upper_edge_hertz = 3800.0
        linear_to_mel_weight_matrix = \
            tf.contrib.signal.linear_to_mel_weight_matrix(
                num_mel_bins=N_MELS,
                num_spectrogram_bins=spectrogram_.shape[-1].value, 
                sample_rate=SAMPLE_RATE,
    #                 lower_edge_hertz=lower_edge_hertz,
    #                 upper_edge_hertz=upper_edge_hertz
            )
        mel_spectrograms_ = tf.tensordot(spectrogram_,
                                        linear_to_mel_weight_matrix,1)
        mel_spectrograms_.set_shape(spectrogram_.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms_ = tf.log(mel_spectrograms_ + 1e-6)
        # log_mel_spectrograms_ = raw_audio2log_mel_spec_op(raw_audio_input, SAMPLE_RATE, FFT_S,HOP_S, N_MELS)
        return log_mel_spectrograms_
        
    mel_spec = Lambda(extra_spec, name='Feature_Extract')(raw_audio_input)
    softmax_output = model_obj.predict_model(mel_spec)

    tflite_model = Model(raw_audio_input,softmax_output)
    tflite_model.save(predict_model_rawinput)

os.environ["CUDA_VISIBLE_DEVICES"] = ''
converter = tf.lite.TFLiteConverter.from_keras_model_file(predict_model_rawinput,input_shapes = {"padded_datas" : [None,80,128]})
tflite_model = converter.convert()
with open(converted_predict_model_tflite_path, "wb") as f:
    f.write(tflite_model)


