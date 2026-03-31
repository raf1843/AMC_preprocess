import numpy as np
import h5py, sys, os
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LRG
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import config

classifier_ = config.CLASSIFIER

if config.CUDA:
    import so_cupy as so
else:
    import so_numpy as so

para = {'figure.figsize'  : (8, 6) }
plt.rcParams.update(para)

mod = range(19)
trans_list = ['SCD', 'CHTC', 'CCSD']
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
    
save_dir = f"{config.TRANSFERSET_NAME}-{classifier_}"


###############################################################
############################# train and test functions
###############################################################
def sys_out(msg):
    print (msg)
    with open(f"../{save_dir}/logbook_second_trans.txt", 'a') as out:
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


def run(snr, trans, profile):
    train = []
    test = []
    for i in mod:
        for j in snr:
            print(f"Processing {mod_list[i]} + {j} SNR")
            base = i * 26 + j
            s = x_train[base,:,:]
            s = getattr(so, trans)(s=s)
            if profile:
                s = block_reduce(s, block_size=(1, 1, 64), func=np.max).reshape((tr, -1))
            else:
                s = s.reshape((tr, -1))
            train.append(s)

            s = x_test[base,:,:]
            s = getattr(so, trans)(s=s)
            if profile:
                s = block_reduce(s, block_size=(1, 1, 64), func=np.max).reshape((te, -1))
            else:
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


def testrun(trans, s, profile):
    test = []
    s = getattr(so, trans)(s=s)
    if profile:
        s = block_reduce(s, block_size=(1, 1, 64), func=np.max).reshape((config.TEST_SIZE, -1))
    else:
        s = s.reshape((config.TEST_SIZE, -1))
    test.append(s)
    test = np.asarray(test)

    if np.iscomplexobj(test):
        test = np.abs(test)

    test = test.reshape((-1, test.shape[-1]))
    return test


def train_test(T, profile, snr_range="high"):
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

    x_tr, x_te = run(snr, T, profile)
    yy_tr = create_label(tr * len(snr))
    yy_te = create_label(te * len(snr))

    sys_out(f'start test {snr_range} snr , transform is {T}, profile {profile}')
    sc, cm = classifier(x_tr, yy_tr, x_te, yy_te)
    sys_out('the acc : %f' % sc)
    fig = plt.figure()
    name = f"{snr_range}SNR_{T}_profile{profile}"
    plt.title(name, fontsize = 10)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(cm, xticklabels=mod_list, yticklabels=mod_list, cmap='Greens')
    plt.tight_layout()
    save_path = f"../{save_dir}/{T}/{T}_profile{profile}_cm_{snr_range}SNR"
    fig.savefig(f"{save_path}.png")

    df_cm = pd.DataFrame(cm, columns=mod_list, index=mod_list)
    df_cm.to_csv(f"{save_path}.csv", index=True)

    # Transfer test
    hf = h5py.File(config.TRANSFERSET_SUBSET_PATH, 'r+')
    x_trans_test = hf['test'][:, :]
    hf.close()

    x_trans_te = testrun(T, x_trans_test, profile)
    yy_trans_te = create_label(te, config.TRANSFERSET_LABEL)

    sys_out(f'start transfer test {snr_range} snr , transform is {T}')
    sc, cm = classifier(x_tr, yy_tr, x_trans_te, yy_trans_te)
    sys_out('the acc : %f' % sc)
    fig = plt.figure()
    name = f"{config.TRANSFERSET_NAME}_{snr_range}SNR_{T}_profile{profile}"
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
def main():
    os.makedirs(f"../{save_dir}", exist_ok=True)

    with open(f"../{save_dir}/logbook_second_trans.txt", 'a') as out:
        out.write("Start \n")   
    
    #for tt in trans_list:
        #sys_out("start {} graph train and test".format(tt))
        #train_test(tt, False)
        #train_test(tt, False, "med")
        #train_test(tt, False, "low")


    for tt in trans_list:
        sys_out("start {} profile train and test".format(tt))
        train_test(tt, True)
        train_test(tt, True, "med")
        train_test(tt, True, "low")

    sys_out('DONE')

if __name__ == '__main__':
    main()
