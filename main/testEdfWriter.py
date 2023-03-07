import os
from datetime import datetime
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
# 只截取60s的数据
raw.crop(tmax=60).load_data()
my_annot = mne.Annotations(onset=[3, 5, 7],
                           duration=[1, 0.5, 0.25],
                           description=['AAA', 'BBB', 'CCC'])
print(my_annot)

raw.set_annotations(my_annot)
print(raw.annotations)

# 构建事件戳
meas_date = raw.info['meas_date'][0] + raw.info['meas_date'][1] / 1e6
orig_time = raw.annotations.orig_time
print(meas_date == orig_time)

