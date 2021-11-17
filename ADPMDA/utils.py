import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl


def load_data(directory, random_seed):

    ID = np.loadtxt(directory + '/HMDD V3.0/SD.txt')
    IM = np.loadtxt(directory + '/HMDD V3.0/SM.txt')
    all_associations = pd.read_csv(directory + '/HMDD V3.0/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
    M_Seq = np.loadtxt(directory + '/mirSeq.txt')
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    samples = sample_df.values

    return ID, IM, M_Seq, samples


def build_graph(directory, random_seed):
    ID, IMS, M_Seq, samples = load_data(directory, random_seed)

    IM = np.hstack((IMS, M_Seq))
    g = dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_sim'] = d_sim

    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim

    disease_ids = list(range(1, ID.shape[0]+1))
    mirna_ids = list(range(1, IM.shape[0]+1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.readonly()

    return g, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc='lower right')

    plt.savefig(directory+'/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curves')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_dif_auc(fprs, tprs, auc, fprs1, tprs1, auc1, fprs2, tprs2, auc2, fprs3, tprs3, auc3, directory, name):

    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    plt.figure(figsize=(20, 8), dpi=600)
    p1 = plt.subplot(1, 2, 1)
    p2 = plt.subplot(1, 2, 2)

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    p1.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Seq+Att (AUC = %.4f)' % mean_auc)
    p2.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Seq+Att (AUC = %.4f)' % mean_auc)

    mean_fpr1 = np.linspace(0, 1, 20000)
    tpr1 = []

    for i in range(len(fprs1)):
        tpr1.append(interp(mean_fpr1, fprs1[i], tprs1[i]))
        tpr1[-1][0] = 0.0

    mean_tpr1 = np.mean(tpr1, axis=0)
    mean_tpr1[-1] = 1.0
    mean_auc1 = np.mean(auc1)
    auc_std1 = np.std(auc1)
    p1.plot(mean_fpr1, mean_tpr1, color='lightcoral', alpha=0.9, linestyle='--', label='NoSeq+NonAtt (AUC = %.4f)' % mean_auc1)
    p2.plot(mean_fpr1, mean_tpr1, color='lightcoral', alpha=0.9, linestyle='--',
            label='NoSeq+NonAtt (AUC = %.4f)' % mean_auc1)

    mean_fpr2 = np.linspace(0, 1, 20000)
    tpr2 = []

    for i in range(len(fprs2)):
        tpr2.append(interp(mean_fpr2, fprs2[i], tprs2[i]))
        tpr2[-1][0] = 0.0

    mean_tpr2 = np.mean(tpr2, axis=0)
    mean_tpr2[-1] = 1.0
    mean_auc2 = np.mean(auc2)
    auc_std2 = np.std(auc2)
    p1.plot(mean_fpr2, mean_tpr2, color='bisque', alpha=0.9, linestyle='--', label='Seq+NonAtt (AUC = %.4f)' % mean_auc2)
    p2.plot(mean_fpr2, mean_tpr2, color='bisque', alpha=0.9, linestyle='--',
            label='Seq+NonAtt (AUC = %.4f)' % mean_auc2)

    mean_fpr3 = np.linspace(0, 1, 20000)
    tpr3 = []

    for i in range(len(fprs3)):
        tpr3.append(interp(mean_fpr3, fprs3[i], tprs3[i]))
        tpr3[-1][0] = 0.0

    mean_tpr3 = np.mean(tpr3, axis=0)
    mean_tpr3[-1] = 1.0
    mean_auc3 = np.mean(auc3)
    auc_std3 = np.std(auc3)
    p1.plot(mean_fpr3, mean_tpr3, color='lightblue', alpha=0.9, linestyle='--', label='NoSeq+Att (AUC = %.4f)' % mean_auc3)
    p2.plot(mean_fpr3, mean_tpr3, color='lightblue', alpha=0.9, linestyle='--',
            label='NoSeq+Att (AUC = %.4f)' % mean_auc3)

    p1.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    p1.axis([-0.05, 1.05, -0.05, 1.05])
    p1.set_xlabel('False Positive Rate')
    p1.set_ylabel('True Positive Rate')
    p1.legend(loc='lower right')

    p2.axis([0.05, 0.3, 0.8, 1.0])
    p2.set_xlabel('False Positive Rate')
    p2.set_ylabel('True Positive Rate')
    p2.legend(loc='upper right')
    plt.savefig(directory + '/%s.jpg' % name, dpi=600, bbox_inches='tight')
    plt.close()