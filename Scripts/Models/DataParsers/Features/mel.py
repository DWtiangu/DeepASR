import librosa
import numpy as np
def _parse_one_audio_melspec(au_data_obj, fft_s, hop_s, target_sr, n_mels ):
    """
    ori_y, ori_sr:应当由librosa.load读取的原始音频数据ori_y以及音频采样率ori_sr
    fft_s:一个短时傅里叶变换的窗长，单位为秒
    hop_s：窗之间间隔长，单位为秒

    返回：行数为 duration_s(音频长度,秒)//hop_s, 列数为 n_mels 的二维数组，纵向代表时间(一行跨越 hop_s 秒)，横向代表频率（物理范围：0-(target_sr//2) Hz,一列跨越fft_s Hz），大小代表能量（单位：db）
    """
    ori_y, ori_sr = au_data_obj.get_data()
    y = librosa.resample(ori_y, ori_sr, target_sr)
    sr = target_sr

    n_fft = round(fft_s * sr)
    hop_length = round(hop_s * sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=None, window='hann', center=False))
    powerS = S**2
    mel_powerS = librosa.feature.melspectrogram(S=powerS,n_mels = n_mels)
    mel_powerS_db = librosa.power_to_db(mel_powerS)

    return mel_powerS_db.T

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
def _parse_one_audio_tf_melspec(au_data_obj,fft_s, hop_s, target_sr, n_mels ):
    n_fft = round(fft_s * target_sr)
    hop_length = round(hop_s * target_sr)
    
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=target_sr)
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
        mel_spectrograms_ = tf.tensordot(spectrogram_,
                                        linear_to_mel_weight_matrix, 1)
        mel_spectrograms_.set_shape(spectrogram_.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms_ = tf.log(mel_spectrograms_ + 1e-6)
        res = sess.run(
            log_mel_spectrograms_,
            feed_dict={
                wav_filename_placeholder: au_data_obj.filename
            })
        
        return res