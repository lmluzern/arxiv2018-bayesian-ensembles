'''
Created on April 27, 2018

@author: Edwin Simpson
'''
import os

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon')
regen_data = False

gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
    load_data.load_biomedical_data(regen_data)#, debug_subset_size=50000)
#
# this is the one we used in the paper, result_started-2019-08-22-06-17-54-Nseen56858.csv
best_nu0factor = 1
best_diags = 10
best_factor = 100
best_outside_factor = 1

# ------------------------------------------------------------------------------------------------
exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, outside_factor=best_outside_factor)

exp.save_results = True
exp.opt_hyper = False

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor

# # run all the methods that don't require tuning here
exp.methods =  [
                'bac_seq_integrateIF',
]

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                 new_data=regen_data
                 )

# ------------------------------------------------------------------------------------------------
# this is the one we used in the paper, result_started-2019-08-22-06-17-54-Nseen56858.csv
best_nu0factor = 1
best_diags = 10
best_factor = 100
best_outside_factor = 10

exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, outside_factor=best_outside_factor)

exp.save_results = True
exp.opt_hyper = False

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor

# # run all the methods that don't require tuning here
exp.methods =  [
                'bac_seq_integrateIF',
]

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                 new_data=regen_data
                 )
#
# # ------------------------------------------------------------------------------------------------
# nu_factors = [0.01, 0.1, 1]
# diags = [0.1, 1, 10]
# factors = [0.1, 1, 10]
#
# methods_to_tune = [
#                    'bac_seq_integrateIF',
#                    ]
#
# best_bac_wm_score = -np.inf
#
# # tune with small dataset to save time
# idxs = np.argwhere(gt_task1_dev != -1)[:, 0]
# idxs = np.concatenate((idxs, np.argwhere(gt != -1)[:, 0]))
# idxs = np.concatenate((idxs, np.argwhere((gt == -1) & (gt_task1_dev == -1))[:300, 0]))  # 100 more to make sure the dataset is reasonable size
#
# tune_gt_dev = gt_task1_dev[idxs]
# tune_annos = annos[idxs]
# tune_doc_start = doc_start[idxs]
# tune_text = text[idxs]

#for m, method in enumerate(methods_to_tune):
#    print('TUNING %s' % method)
#
#    best_scores = exp.tune_alpha0(diags, factors, nu_factors, method, tune_annos, tune_gt_dev, tune_doc_start,
#                                  output_dir, tune_text, metric_idx_to_optimise=11)
#    best_idxs = best_scores[1:].astype(int)
#    exp.nu0_factor = nu_factors[best_idxs[0]]
#    exp.alpha0_diags = diags[best_idxs[1]]
#    exp.alpha0_factor = factors[best_idxs[2]]
#
#    print('Best values: %f, %f, %f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
#
#    # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
#    exp.methods = [method]
#    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=True,
#                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                new_data=regen_data
#    )

# # run all the methods that don't require tuning here
# exp.methods =  [
#                 'bac_seq_integrateIF_then_LSTM',
#                 'bac_seq_integrateIF_integrateLSTM_atEnd',
# ]
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                 new_data=regen_data
#                 )
