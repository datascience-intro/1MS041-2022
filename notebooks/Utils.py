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
    import numpy as np
    data = np.array(data_sequence)
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

def plotEDF(numRelFreqPairs, force_display=True):
    import numpy as np
    if type(numRelFreqPairs) == list:
        numRelFreqPairs = np.array(numRelFreqPairs)

    (keys,counts) = (numRelFreqPairs[:,0],numRelFreqPairs[:,1])
    frequencies = counts/np.sum(counts)
    emf = np.stack([keys,frequencies],axis=-1)
    cumFreqs = np.cumsum(frequencies)
    edf = np.stack([keys,cumFreqs],axis=-1)

    #Plotting using matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5,5))
    #plt.gca().spines['bottom'].set_position('zero')
    #plt.gca().spines['left'].set_position('zero')
    #plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['right'].set_visible(False)

    plt.scatter(keys,cumFreqs)
    plt.hlines(cumFreqs[:-1],keys[:-1],keys[1:])
    plt.vlines(keys[1:],cumFreqs[:-1],cumFreqs[1:],linestyle=':')
    #plt.step(keys,cumFreqs,where='post')

    #Title
    plt.title("Empirical Distribution Function")

    if (force_display):
        # Force displaying
        plt.show()
