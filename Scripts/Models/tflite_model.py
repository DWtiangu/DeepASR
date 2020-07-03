from .AcousticModel import AcousticModel
from config import mel_fea

class CNN1d_CTC_PinYin_Sample_4tflite(AcousticModel):
    import tensorflow as tf
    from tensorflow.python.ops import io_ops
    from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
    def main_structure(self,input_layername='filename'):
        '''input_keras_layer -> softmax_out_layer'''
        assert self.LabelParser.label_type == 'pinyin'

        from keras.models import Sequential, Model
        from keras.layers import Dense, Dropout, Input, BatchNormalization ,Activation
        from keras.layers import Conv1D, MaxPooling1D
        
        fft_s = mel_fea['kwargs']['fft_s'] # fft_s:一个短时傅里叶变换的窗长，单位为秒
        hop_s = mel_fea['kwargs']['hop_s']  # hop_s：窗之间间隔长，单位为秒
        target_sr = mel_fea['kwargs']['target_sr'] # 统一音频采样率目标，音频将自动重采样为该采样率
        n_mels = mel_fea['kwargs']['n_mels'] # mel 特征维度

        wav_filename_placeholder_ = tf.placeholder(
            tf.string, [], name='filename')
        wav_loader = io_ops.read_file(wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=target_sr)

        n_fft = round(fft_s * target_sr)
        hop_length = round(hop_s * target_sr)
        stfts_ = tf.contrib.signal.stft(
            wav_decoder.audio,
            frame_length=n_fft,
            frame_step=hop_length,
            fft_length=None)
        spectrogram_ = tf.abs(stfts_)
        lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
        linear_to_mel_weight_matrix = \
            tf.contrib.signal.linear_to_mel_weight_matrix(
                num_mel_bins=n_mels,
                num_spectrogram_bins=spectrogram_.shape[-1].value, 
                sample_rate=target_sr,
                lower_edge_hertz=lower_edge_hertz,
                upper_edge_hertz=upper_edge_hertz
            )
        mel_spectrograms = tf.tensordot(spectrogram_,
                                        linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrogram_.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

        x = Conv1D(filters=96,kernel_size=11, padding="same",activation="relu", kernel_initializer="he_normal")(input_data)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(256, 5, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Conv1D(384, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = Conv1D(256, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D()(x)
        x = Dense(2048, activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.LabelParser.LABEL_NUM)(x)
        softmax_out = Activation('softmax', name='softmax_out')(x)
        
        return input_data,softmax_out