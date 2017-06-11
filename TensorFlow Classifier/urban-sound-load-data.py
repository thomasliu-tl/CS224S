parent_dir = 'Sound-Data'
tr_sub_dirs = ['fold1','fold2']
ts_sub_dirs = ['fold3']
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)

tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)