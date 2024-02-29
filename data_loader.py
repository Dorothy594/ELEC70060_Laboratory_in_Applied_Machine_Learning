import numpy as np
# import matplotlib.pyplot as plt
# sample = np.loadtxt('./data/wood_chair/0.txt')
# import sounddevice as sd
# sd.play(sample[:2500]/np.max(np.abs(sample)), samplerate=4000)
# sd.wait()

basedir = './data'
cls_name_by_material = ['glass', 'metal', 'paper', 'plastic', 'rubber', 'wood']
cls_name_to_category = {
    'glass':    0,
    'metal':    1,
    'paper':  2,
    'plastic':     3,
    'rubber':    4,
    'wood':   5
}
# cls_name_by_object = []
with open(f'{basedir}/objects_name.txt') as names:
    cls_name_by_object = list(map(lambda name: name.strip('\n'), names.readlines()))

num_object = len(cls_name_by_object)
num_sample_per_object = 20
num_sample = num_object * num_sample_per_object
sample_len = 4000
bound = 32768.


def lab_data_loader():
    # audio sample collected in lab
    sample = np.zeros((num_sample, sample_len))
    # label of material category and object category
    label = np.zeros((num_sample, 2))
    idx = 0
    for i, object_name in enumerate(cls_name_by_object):
        for _ in range(num_sample_per_object):
            sample[idx, :] = np.loadtxt(f'{basedir}/{object_name}/{idx % 20}.txt') / bound
            label[idx, :] = np.array([cls_name_to_category[object_name.split('_')[0]], i])
            idx += 1
    return sample, label


if __name__ == '__main__':
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from scipy import signal

    pca = PCA(n_components=2)

    sample, label = lab_data_loader()
    print(f'shape of sample: {sample.shape}')
    print(f'shape of label: {label.shape}')
    pca.fit(sample)
    sample_pca_embedding = pca.transform(sample)

    for i in range(6):
        plt.scatter(sample_pca_embedding[i*100:i*100+100, 0], sample_pca_embedding[i*100:i*100+100, 1])
    plt.legend(cls_name_by_material)
    plt.title('Raw data reduce to 2-dimensional with PCA')
    plt.show()

    b, a = signal.butter(8, [0.1, 0.8], btype='bandpass')
    filtered_sample = signal.filtfilt(b, a, sample[:, :2000])
    fft_filtered_sample = np.abs(np.fft.fft(filtered_sample))[:, :1000]
    pca.fit(fft_filtered_sample)
    embedding = pca.transform(fft_filtered_sample)
    for i in range(6):
        plt.scatter(embedding[i*100:i*100+100, 0], embedding[i*100:i*100+100, 1])
    plt.legend(cls_name_by_material)
    plt.title('Pre-processed data reduce to 2-dimensional with PCA')
    plt.show()