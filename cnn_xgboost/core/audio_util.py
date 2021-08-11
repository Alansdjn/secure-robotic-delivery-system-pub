import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import time
import os
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
import shutil

class AudioUtil:
    
    @staticmethod
    def load(path):
        data, sr = librosa.load(path, sr=None)
        return data, sr
    
    @staticmethod
    def resample(src_path, dest_path, resample_rate=16000):
        """resample the audio using the given sr, return the file path"""
        if resample_rate == None:
            raise Exception('must input resample rate')
        if src_path == None:
            raise Exception('must input source file path')
        if dest_path == None:
            raise Exception('must input resample file path')
        
        # load data
        y, sr = AudioUtil.load(src_path)
        if sr != resample_rate:
            y = librosa.resample(y, sr, resample_rate)

        # save the resample file
        sf.write(dest_path, y, resample_rate)
        
        return dest_path

    @staticmethod
    def resample_16k(data, sr):
        return data if sr == 16000 else librosa.resample(data, sr, 16000)

    @staticmethod
    def arecord(file, duration=2):
        """Used to record an audio"""

        # default file name is digit
        if file == None:
            file = 'digit.wav'

        # default duration is 2 seconds
        if duration == None:
            duration = 2
        
        # Delay for 1 second, give the user enough time
        # time.sleep(1)
        
        # record a audio, rate only support 44100
        cmd = 'arecord --device=hw:1,0 --format S16_LE --rate 44100 -V mono -d ' + str(duration) + ' -c1 ' + file
        os.system(cmd)
        print()
        
        # Since the rate is 44100, we need resample it to 16000
        AudioUtil.resample(file, file, 16000)
        
        return file
    
    @staticmethod
    def shift_center(data):
        sum_head, sum_tail = 0, 0
        i, j = 0, len(data) - 1
        while i < j:
            if sum_head <= sum_tail:
                sum_head += abs(data[i])
                i += 1
            else:
                sum_tail += abs(data[j])
                j -= 1
        shift = int(len(data) // 2) - i

        return data if shift == 0 else np.roll(data, shift)
    
    @staticmethod
    def resize(data, sr=16000, time_length=1):
        """time_length in second"""
        max_length = int(time_length * sr)
        data_length = len(data)

        if data_length == max_length:
            return data
        
        half = int(abs(data_length - max_length) // 2)
        if data_length > max_length:
            return data[half:half+max_length]
        
        padding_head = np.zeros(half)
        padding_tail = np.zeros(max_length - len(data) - half)
        return np.concatenate((padding_head, data, padding_tail), axis=None)

    @staticmethod
    def split_audio(file, min_silence_len=300, silence_thresh=-36, keep_silence=200, seek_step=1):
        if not file.endswith('wav'):
            raise Exception('must input a wav file')

        data = AudioSegment.from_wav(file)
        audio_chunks = split_on_silence(data,
            # must be silent for at least xxx a minsecond
            min_silence_len,
            # consider it silent if quieter than -16 dBFS
            silence_thresh,
            # leave some silence at the beginning and end of the chunks.
            keep_silence,
            # step size for interating over the segment in ms
            seek_step
        )

        # delete all files in folder split_audio
        split_audio_root = os.path.join(os.path.dirname(os.path.abspath(file)), 'split_audio')
        if os.path.isdir(split_audio_root):
            shutil.rmtree(split_audio_root)
        os.makedirs(split_audio_root)
        print('Save split audio files to: [%s]' % split_audio_root)

        out_files = []
        for i, chunk in enumerate(audio_chunks):
            out_file = "{}/chunk{}.wav".format(split_audio_root, i)
            chunk.export(out_file, format="wav")
            out_files.append(out_file)
        
        return out_files

    @staticmethod
    def split_4_segments(file, min_silence_len=300, keep_silence=200, seek_step=1):
        if not file.endswith('wav'):
            raise Exception('must input a wav file')

        data = AudioSegment.from_wav(file)
        thresh_candidates = []

        # The audio is 16bits, [0 ~ -96dBFS]
        silence_thresh = -20
        while silence_thresh > -50:
            audio_segments = split_on_silence(data,
                # must be silent for at least xxx a minsecond
                min_silence_len,
                # consider it silent if quieter than -16 dBFS
                silence_thresh,
                # leave some silence at the beginning and end of the chunks.
                keep_silence,
                # step size for interating over the segment in ms
                seek_step)

            if len(audio_segments) == 4:
                thresh_candidates += [silence_thresh]
            silence_thresh += -1

        if len(thresh_candidates) > 0:
            silence_thresh = thresh_candidates[int(len(thresh_candidates) // 2)]
            audio_segments = AudioUtil.split_audio(file, min_silence_len, silence_thresh, keep_silence, seek_step)

        if len(audio_segments) != 4:
            raise Exception("Cann't split the audio into 4 segments.")

        return audio_segments

    @staticmethod
    def waveplot(files, pins):
        fig, ax = plt.subplots(nrows=len(files), sharey=True, sharex=True)
        for i, file in enumerate(files):
            y, sr = AudioUtil.load(file)
            librosa.display.waveplot(y, sr=sr, ax=ax[i])

        split_audio_folder = os.path.dirname(os.path.abspath(files[0]))
        out_file = os.path.join(split_audio_folder, "waveplot_{}.png".format(''.join(pins)))
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        return out_file







