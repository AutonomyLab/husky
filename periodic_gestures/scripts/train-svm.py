#! /usr/bin/env python




from sklearn import svm
from sklearn import cross_validation
import sys, os, numpy as np, cPickle as pickle



DIR = None
OUTFILE = None
MODE = None # either "aggregate" or "flat", representing a time-indexed flat motion descriptor, or an aggregated sum descriptor

DATA_INFILE = None
DATA_OUTFILE = None

TEST_SIZE = 0.4 # reserve 40% of our data as final verification data

CROSS_VALIDATION_FOLDS = 5

KERNEL = None
C_VALUE = None

#------------------------------------------------------------------------------


def train_svm():
    posdir = "%s%spositive" % (DIR,os.sep)
    negdir = "%s%snegative" % (DIR,os.sep)

    # training data
    X = None

    # training data classes
    Y = None

    if DATA_INFILE == None:

        print("Loading motion descriptors from %s" % DIR)

        num = len(os.listdir(posdir)) + len(os.listdir(negdir))
        i = 0
        # load positive examples
        for sample_dir in os.listdir(posdir):
            if MODE == "aggregate":
                sample_file = "%s%s%s%saggregate.pickle" % (posdir, os.sep, sample_dir, os.sep)
            elif MODE == "flat":
                sample_file = "%s%s%s%sflattened.pickle" % (posdir, os.sep, sample_dir, os.sep)

            with open(sample_file) as fp:
                sample = pickle.load(fp)
                flattened = np.ndarray.flatten(sample)
                if X == None:
                    print("Allocating space for %d floats" % (num*len(flattened)))
                    X = np.zeros((num, len(flattened)))
                    Y = np.zeros(num, np.int)
                X[i] = flattened
                Y[i] = 1
                i += 1

        # load negative examples
        for sample_dir in os.listdir(negdir):
            if MODE == "aggregate":
                sample_file = "%s%s%s%saggregate.pickle" % (negdir, os.sep, sample_dir, os.sep)
            elif MODE == "flat":
                sample_file = "%s%s%s%sflattened.pickle" % (negdir, os.sep, sample_dir, os.sep)

            with open(sample_file) as fp:
                sample = pickle.load(fp)
                X[i] = np.ndarray.flatten(sample)
                Y[i] = 0
                i += 1


    else:
        print("Loading pre-saved motion descriptors from %s" % DATA_INFILE)

        with open(DATA_INFILE) as fp:
            in_data = pickle.load(fp)
            X = in_data[0]
            Y = in_data[1]

    numpos = np.sum(Y)
    numneg = len(Y) - numpos

    print("Finished loading data. Shape: %s by %s" % (len(X), len(X[0])))
    print("Num positive: %s" % numpos)
    print("Num negative: %s" % numneg)

    if DATA_OUTFILE != None:
        print("Saving bundled motion descriptors to %s" % DATA_OUTFILE)

        with open(DATA_OUTFILE, "wb") as fp:
            out_data = (X,Y)
            pickle.dump(out_data, fp)


    trained_svm = cross_validate(X,Y)

    print("Done training, dumping SVM to %s" % OUTFILE)
    with open(OUTFILE, "wb") as fp:
        pickle.dump(trained_svm, fp)


#------------------------------------------------------------------------------

def cross_validate(X, Y):
    print("Training SVM")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE)
    print "Test: total length: %s" % len(X)
    print "Test: len(X_train): %s" % len(X_train)
    print "Test: len(X_test): %s" % len(X_test)




# -------------- CHOOSING KERNEL -----------------


    best_kernel = None
    best_score = 0
    best_std = 0
    best_svm = None

    if KERNEL == None:
        kernels = ["linear", "poly", "rbf", "sigmoid"]
    else:
        kernels = [KERNEL]
    kernel_scores = [0]*len(kernels)
    kernel_stds = [0]*len(kernels)
    for k in range(0,len(kernels)):
        kernel = kernels[k]
        print "Training SVM with %s kernel" % kernel

        cv_svm = svm.SVC(kernel=kernel, scale_C=True)
        scores = cross_validation.cross_val_score(cv_svm, X_train, Y_train, cv=CROSS_VALIDATION_FOLDS)

        print "Kernel %s accuracy: %f (+/- %f)" % (kernel, scores.mean(), scores.std() * 2)

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_std = scores.std()
            best_kernel = kernel
        kernel_scores[k] = scores.mean()
        kernel_stds[k] = scores.std()

    print "Best kernel: %s with score: %f (+/- %f)" % (kernel, scores.mean(), scores.std() * 2)



# ---------- CHOOSING BOX CONSTRAINT C -----------------------

    best_score = 0
    best_std = 0
    best_c = 0

    if C_VALUE == None:
        c_values = [0.01, 0.1, 0.5, 1.0, 5.0, 25.0, 100.0]
    else:
        c_values = [C_VALUE]
    c_scores = [0]*len(c_values)
    c_stds = [0]*len(c_values)
    for i,C in enumerate(c_values):
        print "Training %s SVM with C=%s" % (best_kernel, C)
        cv_svm = svm.SVC(kernel=best_kernel, C=C, scale_C=True)
        scores = cross_validation.cross_val_score(cv_svm, X_train, Y_train, cv=CROSS_VALIDATION_FOLDS)
        print "%s SVM with C=%s accuracy: %f (+/- %f)" % (best_kernel, C, scores.mean(), scores.std() * 2)

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_std = scores.std()
            best_c = C
            best_svm = cv_svm
        c_scores[i] = scores.mean()
        c_stds[i] = scores.std()

    print "Best C: %s with score: %f (+/- %f)" % (best_c, best_score, best_std)

#------------------------------------------------------------------------------

    print "Training final SVM"
    best_svm = svm.SVC(kernel=best_kernel, C=best_c, scale_C=True)
    best_svm.fit(X_train, Y_train)
    score = best_svm.score(X_test, Y_test)

    print "Best SVM score on test data: %s" % score

    return best_svm

#------------------------------------------------------------------------------

def train_test_split(X, Y, test_size=0.4):
    test_num = int(len(Y) * test_size)
    indices = np.random.permutation(test_num)

    X_test = X[indices]
    Y_test = Y[indices]

    X_train = np.delete(X, indices, axis=0)
    Y_train = np.delete(Y, indices)

    return (X_train, X_test, Y_train, Y_test)

#------------------------------------------------------------------------------

if __name__ == "__main__":
    for i,arg in enumerate(sys.argv):
        if arg == "--dir":
            DIR = os.path.normpath(sys.argv[i+1])
        elif arg == "--outfile":
            OUTFILE = os.path.normpath(sys.argv[i+1])
        elif arg == "--mode" and sys.argv[i+1] == "aggregate":
            MODE = "aggregate"
        elif arg == "--mode" and sys.argv[i+1] == "flat":
            MODE = "flat"
        elif arg == "--data-in":
            DATA_INFILE = os.path.normpath(sys.argv[i+1])
        elif arg == "--data-out":
            DATA_OUTFILE = os.path.normpath(sys.argv[i+1])
        elif arg == "--kernel":
            KERNEL = sys.argv[i+1]
        elif arg == "--c-value":
            C_VALUE = float(sys.argv[i+1])
        elif arg == "--cv-folds":
            CROSS_VALIDATION_FOLDS = int(sys.argv[i+1])


    if (DIR == None and DATA_INFILE == None) or OUTFILE == None or MODE == None:
        print "usage: %s [ --dir <dir-with-svm-input> | --data-in <dir-with-presaved-input-bundle> ] --data-out <save-input-bundle-to-here> --outfile <trained-svm-file> --mode <aggregate|flat>" % sys.argv[0]
        sys.exit(1)


    train_svm()
