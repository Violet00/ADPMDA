import warnings

from train import Train
from utils import plot_auc_curves, plot_prc_curves, plot_dif_auc


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    fprs, tprs, auc, precisions, recalls, prc = Train(directory='data',
                                                      epochs=1000,
                                                      k=10,
                                                      attn_size=64,
                                                      out_dim=128,
                                                      dropout=0.5,
                                                      slope=0.3,
                                                      lr=0.001,
                                                      wd=5e-3,
                                                      random_seed=2021,
                                                      cuda=True,
                                                      model_type='ADPMDA')

    plot_auc_curves(fprs, tprs, auc, directory='roc_result', name='test_auc')
    plot_prc_curves(precisions, recalls, prc, directory='roc_result', name='test_prc')
