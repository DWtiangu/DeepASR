import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

def _parse_one_audio_tf_raw(au_data_obj,fft_s, hop_s, target_sr, n_mels ):
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1,desired_samples=target_sr)
        return sess.run(
            wav_decoder, feed_dict={
                wav_filename_placeholder: au_data_obj.filename
            }).audio.flatten()