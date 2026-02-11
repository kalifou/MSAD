import pandas as pd
import os
import ipdb


def filter_list_channels_evaluated(dir_models="data/benchmark_esa_binarized/m1/test/metrics_working_copy/"):
	list_models = os.listdir(dir_models)

	all_channels = list()
	for model_i in list_models:
	    local_model_roc_logs = dir_models + model_i + "/R_AUC_ROC.csv"
	    model_i_roc_perfs = pd.read_csv(local_model_roc_logs)
	    model_i_list_channels = [ch_i.split(
	        "@Channel_")[1] for ch_i in model_i_roc_perfs['Unnamed: 0']]
	    all_channels.append(model_i_list_channels)

	s_all = set()
	for list_ch_i in all_channels:
	    s_all.update(list_ch_i)

	return list(s_all)


def keep_intersection_of_available_channels(dir_models="data/benchmark_esa_binarized/m1/train/metrics_working_copy/"):
    list_models = os.listdir(dir_models)    
    all_channels = list()

    for model_i in list_models:
        local_model_roc_logs = dir_models + model_i + "/R_AUC_ROC.csv"
        model_i_roc_perfs = pd.read_csv(local_model_roc_logs)
        model_i_list_channels = [ch_c.split("@Channel_")[1] for ch_c in model_i_roc_perfs['Unnamed: 0']]
        #ipdb.set_trace(context=35)
        all_channels.append(model_i_list_channels)
    
    s_all = set(all_channels[0])
    for list_ch_i in all_channels:
        print(s_all)
        s_all = s_all.intersection(set(list_ch_i))

    list_not_included = set()
    for model_i in list_models:
        local_model_roc_logs = dir_models + model_i + "/R_AUC_ROC.csv"
        model_i_roc_perfs = pd.read_csv(local_model_roc_logs)
        print(model_i, len(model_i_roc_perfs['Unnamed: 0']))
        local_df_model_mi = model_i_roc_perfs.copy()
        for idx_c, ch_c in enumerate(model_i_roc_perfs['Unnamed: 0']):
            local_channel_idx = ch_c.split("@Channel_")[1]
            if local_channel_idx not in s_all:
                print(model_i + " - Missing channel: ", local_channel_idx, idx_c)
                list_not_included.update(local_channel_idx)
                #ipdb.set_trace(context=35)
                assert model_i_roc_perfs.iloc[idx_c]["Unnamed: 0"].split("@Channel_")[1] == local_channel_idx
                local_df_model_mi = local_df_model_mi.drop(model_i_roc_perfs.iloc[idx_c].name)
        #ipdb.set_trace(context=35)
        assert len(local_df_model_mi) == len(s_all)
        
        #target_dir = "/tmp/di38jul/metrics_working_copy/" + model_i
        #if not os.path.isdir(target_dir):
        #    os.makedirs(target_dir)
        local_df_model_mi.to_csv(local_model_roc_logs)
                    
    #ipdb.set_trace(context=35)
    for m_i in list_not_included:
        assert m_i not in s_all
    #ipdb.set_trace(context=35)
    
    #return list(s_all)


if __name__ == "__main__":
    keep_intersection_of_available_channels()
