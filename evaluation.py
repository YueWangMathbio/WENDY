
"""
# the following function calculate auroc and aupr to evaluate
# the calculated GRN and the true GRN.
# the input A is the true GRN (has only 0, 1, -1), 
# and the input result is the calculated GRN
# this function requires matching both the direction of the regulation
# and the activation/inhibition of the regulation

# see the SINCERETIES paper Section 3.1 for more details
# https://academic.oup.com/bioinformatics/article/34/2/258/4158033
"""
import sklearn


def directed_evaluation(A, result): # A is true GRN, result is inferred GRN
    n = len(A)
    A = A.reshape(1, n ** 2)[0]
    result = result.reshape(1, n ** 2)[0]
    sorted_result = [[result[i], i] for i in range(n ** 2) if i % (n + 1) != 0]
    sorted_result.sort(key = lambda x: abs(x[0]), reverse = True)
    tot = n ** 2 - n
    nop = sum([A[i] != 0 for i in range(n ** 2)])
    non = tot - nop
    tp = [0] * (tot + 1)
    tn = [non] * (tot + 1)
    fp = [0] * (tot + 1)
    fn = [nop] * (tot + 1)
    for i in range(1, tot + 1):        
        tp[i] = tp[i-1]
        tn[i] = tn[i-1]
        fp[i] = fp[i-1]
        fn[i] = fn[i-1]
        [val, ind] = sorted_result[i-1]
        if val >= 0:
            if A[ind] == 1:
                tp[i] += 1
                fn[i] -= 1
            elif A[ind] == 0:
                fp[i] += 1
                tn[i] -= 1
            else:
                fp[i] += 1
                fn[i] -= 1
        else:
            if A[ind] == -1:
                tp[i] += 1
                fn[i] -= 1
            elif A[ind] == 0:
                fp[i] += 1
                tn[i] -= 1
            else:
                fp[i] += 1
                fn[i] -= 1
    tpr = [0] * (tot+ 1)
    fpr = [0] * (tot+ 1)
    precision = [0] * (tot+ 1)
    recall = [0] * (tot+ 1)
    for i in range(1, tot+ 1):
        if tp[i] + fn[i] > 0:
            tpr[i] = tp[i] / (tp[i] + fn[i])
        if fp[i] + tn[i] > 0:
            fpr[i] = fp[i] / (fp[i] + tn[i])
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])
    
    auroc = sklearn.metrics.auc(fpr, tpr)
    aupr = sklearn.metrics.auc(recall, precision)
    return auroc, aupr # return the AUROC and AUPR values
