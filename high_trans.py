import numpy as np
import h5py, sys, argparse
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LRG
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import os
import config

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', type=str, default='LDA')
parser.add_argument('--cuda', action='store_true')
# could try playing with these
parser.add_argument('--window_size', type=int, default=64)
parser.add_argument('--step', type=int, default=16)
args = parser.parse_args()
classifier_ = args.classifier
window_size = args.window_size
step = args.step

if args.cuda:
    import ho_cupy as ho
else:
    import ho_numpy as ho


para = {'figure.figsize'  : (8, 6) }
plt.rcParams.update(para)

mod = range(19)
trans_list = ['element_HOS', 'RD_CTCF']
mod_list = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK',\
         '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', \
         '256QAM', 'GMSK', 'OQPSK']

hf = h5py.File('201801a_subset.h5', 'r+')
x_test = hf['test'][:, :, :]
x_train = hf['train'][:, :, :]
hf.close()

pts = 300
tr = int(pts // 10) * 9
te = pts - tr

hf = h5py.File(config.TRANSFERSET_SUBSET_PATH, 'r+')
x_trans_test = hf['test'][:, :]
hf.close()

save_dir = f"{config.TRANSFERSET_NAME}-{classifier_}"
os.makedirs(f"../{save_dir}", exist_ok=True)

with open(f"../{save_dir}/logbook_high_trans.txt", "w") as out:
    out.write("Start \n")

###############################################################
############################# train and test functions
###############################################################
def sys_out(msg):
    print (msg)
    with open(f"../{save_dir}/logbook_high_trans.txt", 'a') as out:
        out.write(msg + '\n')


def create_label(num, mods=mod):
    mo = []
    for m in mods:
        mo.append([m] * num)
    mo = np.hstack(mo)
    return mo


def classifier(out_tr, yy_tr, out_te, yy_te):
    if classifier_ == 'LDA':
        lda = LDA().fit(out_tr, yy_tr)
        cm = confusion_matrix(yy_te, lda.predict(out_te), labels=mod)
        return lda.score(out_te, yy_te), cm

    elif classifier_ == 'SGD':
        clf = SGD(alpha=0.1, max_iter=100, shuffle=True, random_state=0, tol=1e-3)
        clf.fit(out_tr, yy_tr)
        cm = confusion_matrix(yy_te, clf.predict(out_te), labels=mod)
        return clf.score(out_te, yy_te), cm
    
    elif classifier_ == 'LRG':
        clf = LRG(random_state=0).fit(out_tr, yy_tr)
        cm = confusion_matrix(yy_te, clf.predict(out_te), labels=mod)
        return clf.score(out_te, yy_te), cm

    else:
        sys.exit(" WRONG CLASSIFIER NAME ! ")


def run(snr, trans):
    train = []
    test = []
    for i in mod:
        for j in snr:
            print(f"Processing {mod_list[i]} + {j} SNR")
            base = i * 26 + j
            s = x_train[base,:,:]
            obj = getattr(ho, trans)
            if isinstance(obj, type):
                obj = obj(window_size, step)
            s = obj(s)
            s = s.reshape((tr, -1))
            train.append(s)

            s = x_test[base,:,:]
            obj = getattr(ho, trans)
            if isinstance(obj, type):
                obj = obj(window_size, step)
            s = obj(s)
            s = s.reshape((te, -1))
            test.append(s)

    train = np.asarray(train)
    test = np.asarray(test)
    l = train.shape[-1]

    if np.iscomplexobj(train):
        train, test = np.abs(train), np.abs(test)
        
    train = train.reshape((-1, l))
    test = test.reshape((-1, l))
    print (train.shape, test.shape)
    return train, test

def testrun(trans):
    test = []
    s = x_trans_test
    obj = getattr(ho, trans)
    if isinstance(obj, type):
        obj = obj(window_size, step)
    s = obj(s)
    s = s.reshape((30, -1))
    test.append(s)
    test = np.asarray(test)

    if np.iscomplexobj(test):
        test = np.abs(test)
    test = test.reshape((-1, test.shape[-1]))
    return test


def train_test(T, snr_range="high"):
    os.makedirs(f"../{save_dir}/{T}", exist_ok=True)

    if snr_range not in ["high", "med", "low"]:
        snr_range = "high"

    match snr_range:
        case "high":
            snr = range(18,26)
        case "med":
            snr = range(8,16)
        case "low":
            snr = range(0,8)

    x_tr, x_te = run(snr, T)
    yy_tr = create_label(tr * len(snr))
    yy_te = create_label(te * len(snr))

    sys_out(f'start test {snr_range} snr , transform is {T}')
    sc, cm = classifier(x_tr, yy_tr, x_te, yy_te)
    sys_out('the acc : %f' % sc)
    fig = plt.figure()
    name = f"{snr_range}SNR_{T}"
    plt.title(name, fontsize = 10)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(cm, xticklabels=mod_list, yticklabels=mod_list, cmap='Greens')
    plt.tight_layout()
    save_path = f"../{save_dir}/{T}/{T}_cm_{snr_range}SNR"
    fig.savefig(f"{save_path}.png")

    df_cm = pd.DataFrame(cm, columns=mod_list, index=mod_list)
    df_cm.to_csv(f"{save_path}.csv", index=True)

    # Transfer test
    x_trans_te = testrun(T)
    yy_trans_te = create_label(te, config.TRANSFERSET_LABEL)

    sys_out(f'start transfer test {snr_range} snr , transform is {T}')
    sc, cm = classifier(x_tr, yy_tr, x_trans_te, yy_trans_te)
    sys_out('the acc : %f' % sc)
    fig = plt.figure()
    name = f"{config.TRANSFERSET_NAME}_{snr_range}SNR_{T}"
    plt.title(name, fontsize = 10)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(cm, xticklabels=mod_list, yticklabels=mod_list, cmap='Blues')
    plt.tight_layout()
    save_path = f"../{save_dir}/{T}/{config.TRANSFERSET_NAME}_{T}_cm_{snr_range}SNR"
    fig.savefig(f"{save_path}.png")

    df_cm = pd.DataFrame(cm, columns=mod_list, index=mod_list)
    df_cm.to_csv(f"{save_path}.csv", index=True)



###############################################################
############################# main function
###############################################################
for tt in trans_list:
    sys_out("start {} train and test".format(tt))
    train_test(tt)
    train_test(tt, "med")
    train_test(tt, "low")

sys_out('DONE')
