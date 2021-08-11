import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import librosa
import librosa.display
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

if __name__ == '__main__':
    from audio_util import AudioUtil
elif __name__ == 'core.data_preprocess':
    from core.audio_util import AudioUtil
elif __name__ == 'cnn_xgboost.core.data_preprocess':
    from cnn_xgboost.core.audio_util import AudioUtil

SR_16K = 16000
DPI = 72

def get_mel_img_data(audio_path, save_path=None, is_gray=False):
    data, sr = AudioUtil.load(audio_path)
    if sr != SR_16K:
        data = AudioUtil.resample_16k(data, sr)
        sr = SR_16K
    # data = AudioUtil.shift_center(data)
    data = AudioUtil.resize(data, sr, time_length=1)

    fig = plt.figure(figsize=[1,1], dpi=DPI, frameon=False)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_frame_on(False)

    # Default values: n_fft=2048, hop_length=512, window='hann'
    S = librosa.feature.melspectrogram(y=data, sr=sr)
    db_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(db_S, x_axis='time', y_axis='mel')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0, dpi=DPI)
    buf.seek(0)

    img = Image.open(buf)
    
    if is_gray == True:
        img = img.convert('L')

    if save_path is not None:
        img.save(save_path)

    # transform=transforms.ToTensor()
    img_tensor = transforms.ToTensor()(img)
    
    buf.close()
    plt.close()

    return img_tensor






