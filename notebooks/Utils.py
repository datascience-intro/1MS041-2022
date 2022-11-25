def basic_stats(data):
    import numpy as np
    from scipy.stats import skew, kurtosis
    print("mean: %.2f\tstd: %.2f\tskew: %.2f\tkurtosis: %.2f" % (np.mean(data),np.std(data),skew(data),kurtosis(data,fisher=False)))

def showURL(url, ht=600):
    """Return an IFrame of the url to show in notebook with height ht"""
    from IPython.display import IFrame
    return IFrame(url, width='95%', height=ht)

def load_sms():
    """
    A wrapper function to load the sms data
    """
    import csv
    lines = []
    hamspam = {'ham': 0, 'spam': 1}
    with open('data/spam.csv', mode='r',encoding='latin-1') as f:
        reader = csv.reader(f)
        # When using the csv reader, each time you use the function
        # next on it, it will spit out a list split at the ','
        header = next(reader)
        # We store this as ("txt",label), where we have used the function
        # hamspam to convert from "ham","spam" to 0 and 1.
        lines = [(line[1],hamspam[line[0]]) for line in reader]

    return lines

def discrete_histogram(data,normed=False):
    import numpy as np
    bins, counts = np.unique(data,return_counts=True)
    width = np.min(np.diff(bins))/4
    import matplotlib.pyplot as plt
    if (normed):
        plt.bar(bins,counts/np.sum(counts),width=width)
    else:
        plt.bar(bins,counts,width=width)

def plotEMF(numRelFreqPairs, force_display = True):
    import matplotlib.pyplot as plt
    import numpy as np
    numRelFreqPairs = np.array(numRelFreqPairs)
    plt.scatter(numRelFreqPairs[:,0],numRelFreqPairs[:,1])
    plt.scatter(numRelFreqPairs[:,0],numRelFreqPairs[:,1])
    for k in numRelFreqPairs:    # for each tuple in the list
        kkey, kheight = k     # unpack tuple
        plt.vlines([kkey],0,kheight,linestyle=':')

    if (force_display):
        plt.show()

def makeFreq(data_sequence):
    """
    Takes a data_sequence in the form of iterable and returns a
    numpy array of the form [keys,counts] where the keys
    are the unique values in data_sequence and counts are how
    many time they appeared
    """
    import numpy as np
    data = np.array(data_sequence)
    if (len(data.shape) == 2):
        (keys,counts) = np.unique(data.T,axis=0,return_counts=True)
        return np.concatenate([keys,counts.reshape(-1,1)],axis=1)
    else:
        (keys,counts) = np.unique(data,return_counts=True)
        return np.stack([keys,counts],axis=-1)

def makeEMF(data_sequence):
    from Utils import makeFreq
    relFreq = makeFreq(data_sequence)
    import numpy as np
    total_sum = np.sum(relFreq[:,1])
    norm_freqs = relFreq[:,1]/total_sum
    return np.stack([relFreq[:,0],norm_freqs],axis=-1)

def makeEDF(data_sequence):
    import numpy as np
    numRelFreqPairs = makeFreq(data_sequence)
    (keys,counts) = (numRelFreqPairs[:,0],numRelFreqPairs[:,1])
    frequencies = counts/np.sum(counts)
    emf = np.stack([keys,frequencies],axis=-1)
    cumFreqs = np.cumsum(frequencies)
    edf = np.stack([keys,cumFreqs],axis=-1)

    return edf

def emfToEdf(emf):
    import numpy as np
    if (type(emf) == list):
        emf = np.array(emf)
    keys = emf[:,0]
    frequencies = emf[:,1]

    cumFreqs = np.cumsum(frequencies)
    edf = np.stack([keys,cumFreqs],axis=-1)
    return edf

def plotEDF(edf,  force_display=True):
    #Plotting using matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5,5))
    #plt.gca().spines['bottom'].set_position('zero')
    #plt.gca().spines['left'].set_position('zero')
    #plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['right'].set_visible(False)

    keys = edf[:,0]
    cumFreqs = edf[:,1]

    plt.scatter(keys,cumFreqs)
    plt.hlines(cumFreqs[:-1],keys[:-1],keys[1:])
    plt.vlines(keys[1:],cumFreqs[:-1],cumFreqs[1:],linestyle=':')
    #plt.step(keys,cumFreqs,where='post')

    #Title
    plt.title("Empirical Distribution Function")

    if (force_display):
        # Force displaying
        plt.show()

def linConGen(m, a, b, x0, n):
    '''A linear congruential sequence generator.

    Param m is the integer modulus to use in the generator.
    Param a is the integer multiplier.
    Param b is the integer increment.
    Param x0 is the integer seed.
    Param n is the integer number of desired pseudo-random numbers.

    Returns a list of n pseudo-random integer modulo m numbers.'''

    x = x0 # the seed
    retValue = [x % m]  # start the list with x=x0
    for i in range(2, n+1, 1):
        x = (a * x + b) % m # the generator, using modular arithmetic
        retValue.append(x) # append the new x to the list
    return retValue

def scatter3d(x,y,z,c=None,size=2,fig=None):
    import plotly.graph_objects as go
    import numpy as np


    if (c == None):
        data = go.Scatter3d(x=x, y=y, z=z,mode='markers',marker=dict(size=size))
        if (fig):
            fig.add_trace(data)
        else:
            fig = go.Figure(data=[data])
    else:
        data = go.Scatter3d(x=x, y=y, z=z,mode='markers',marker=dict(size=size,color=c))
        if (fig):
            fig.add_trace(data)
        else:
            fig = go.Figure(data=[data])
    return fig

def classification_report_interval(
    y_true,
    y_pred,
    labels=None,
    alpha = 0.01,
    union_bound_correction=True
):
    """Produces a classification report with precision, recall and accuracy
    It also uses Hoeffdings inequality to produce confidence intervals around
    each measurement. We can do this with or without multiple measurement
    correction (union bound correction).

    Example output is:
                labels           precision             recall

               0.0  0.88 : [0.50,1.00] 0.40 : [0.15,0.65]
               1.0  0.56 : [0.34,0.78] 0.93 : [0.65,1.00]

          accuracy                                        0.64 : [0.45,0.83]

    Parameters:
    y_true                          -- The true labels
    y_pred                          -- The predicted labels
    labels                          -- TODO
    alpha[0.01]                     -- The confidence level of the intervals
    union_bound_correction[True]    -- If we should compensate with the union bound because we
                                    have multiple intervals to compute in order to keep the level
                                    of confidence for all intervals jointly.

    Returns:
    a printable string.
    """
    import numpy as np
    
    def precision_recall(y_true,
        y_pred,
        labels=None,alpha=0.01, correction=1):
        p = []
        r = []
        f1 = []
        support = []
        for label in labels:
            y_true_pred_label = y_true[y_pred == label]
            precision = np.mean(y_true_pred_label == label)
            delta = (1/np.sqrt(len(y_true_pred_label)))*np.sqrt((1/2)*np.log(2*correction/alpha))
            p.append("%.2f : [%.2f,%.2f]" % (precision, np.maximum(precision-delta,0),np.minimum(precision+delta,1)))

            y_pred_true_label = y_pred[y_true == label]
            recall = np.mean(y_pred_true_label == label)
            delta = (1/np.sqrt(len(y_pred_true_label)))*np.sqrt((1/2)*np.log(2*correction/alpha))
            r.append("%.2f : [%.2f,%.2f]" % (recall, np.maximum(recall-delta,0),np.minimum(recall+delta,1)))

        return (p,r)

    def accuracy_interval(y_true,y_pred,alpha=0.01,correction=1):
        acc = np.mean(y_true == y_pred)
        delta = (1/np.sqrt(len(y_true)))*np.sqrt((1/2)*np.log(2*correction/alpha))
        return "%.2f : [%.2f,%.2f]" % (acc, np.maximum(acc-delta,0),np.minimum(acc+delta,1))

    digits = 18
    target_names = None
    if labels is None:
        labels = list(set(y_true).union(set(y_pred)))
        labels_given = False
    else:
        labels = np.asarray(labels)
        labels_given = True

    target_names = ["%s" % l for l in labels]

    headers = ["precision", "recall"]
    # compute per-class results without averaging
    # Simple correction using the union bound
    # We are computing 2 intervals for each label for precision and recall
    # In addition we are computing 2 intervals for accuracy
    # This is in total 2*n_labels+2
    if (union_bound_correction):
        correction = 2*len(labels)+2
    else:
        correction=1
    p, r = precision_recall(
        y_true,
        y_pred,
        labels=labels,
        alpha=alpha,
        correction=correction
    )

    rows = zip(target_names, p, r)

    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, digits)
    head_fmt = "{:>{width}s} " + " {:>{digits}}" * len(headers)
    report = head_fmt.format("labels", *headers, width=width,digits=digits)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>{digits}s}" * 2 + "\n"
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)
    row_fmt_acc = "{:>{width}s} " + " {:>{digits}s}" * 2 + " {:>{digits}s}""\n"
    report += "\n"
    accuracy = accuracy_interval(y_true,y_pred,alpha=alpha,correction=correction)
    report+=row_fmt_acc.format(*("accuracy","","",accuracy),width=width,digits=digits)

    return report
