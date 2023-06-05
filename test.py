import  numpy as np

a1=np.full((2,2),1)
a2=np.full((2,2),3)
a3=np.full((2,2),7)
a4=np.full((2,2),10)
a5=np.full((2,2),11)

b=[a1,a2,a3,a4,a5]

def diff_season(data,s):
    begin=data[0:s]
    diff_data=[]
    i=len(data)-1
    while(i-s>=0):
        diff_data.insert(0,data[i]-data[i-s])
        i-=1
    return begin,diff_data

def inv_diff_season(begin,diff_data,s):
    data=begin
    max_idx=len(begin)+len(diff_data)
    r=0
    i=0
    while i<len(diff_data):
        for j in range(s):
            data.append(data[s*r+j]+diff_data[i])
            i+=1
            if i>=len(diff_data):break
        r+=1
    return data

def mape(actual, predicted):
    """
    计算 MAPE（平均绝对百分比误差）
    :param actual: 实际值列表
    :param predicted: 预测值列表
    :return: MAPE
    """
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def _update_Es(self, es, alpha, beta, unfold_cores, i, n):

    T_hat = len(unfold_cores)
    begin_idx = self._p + self._q

    As = []
    for t in range(begin_idx, T_hat):
        # t-i
        a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p)], axis=0)
        b = np.sum(
            [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q) if i != j],
            axis=0)
        As.append(unfold_cores[t] - a + b)
    E = np.sum(As, axis=0)
    for t in range(len(es)):
        es[t][i] = self._get_fold_tensor(E / (2 * (begin_idx - T_hat) * beta[i]), n, es[t][i].shape)
    return es

def update_alpha(self,es, alpha, beta,gamma, thet, unfold_cores, i, n,s):
    T_hat=len(unfold_cores)
    begin_idx=s*self._P+s*self._Q
    As= []
    for t in range(begin_idx,T_hat):
        a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p) if i != ii], axis=0)
        b = np.sum(
            [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q) ],axis=0)
        c = np.sum([gamma[ii] * unfold_cores[:t][-(s*ii + 1)] for ii in range(self._P)], axis=0)
        d = np.sum(
            [thet[j] * self._get_unfold_tensor(es[:t][-(s*j + 1)], n) for j in range(self._Q)],axis=0)
        As.append((unfold_cores[t] - a + b-c+d) * unfold_cores[:t][-(i + 1)].T)
    E = np.sum(As, axis=0)

    alpha[i] =np.mean(E /(2 * (T_hat - begin_idx)))
    return alpha

def update_beta(self,es, alpha, beta,gamma, thet, unfold_cores, i, n,s):
    T_hat=len(unfold_cores)
    begin_idx=s*self._P+s*self._Q
    As= []
    for t in range(begin_idx,T_hat):
        a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p) ], axis=0)
        b = np.sum(
            [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q) if i != j],axis=0)
        c = np.sum([gamma[ii] * unfold_cores[:t][-(s*ii + 1)] for ii in range(self._P) ], axis=0)
        d = np.sum(
            [thet[j] * self._get_unfold_tensor(es[:t][-(s*j + 1)], n) for j in range(self._Q)],axis=0)
        As.append((unfold_cores[t] - a + b-c+d) * self._get_unfold_tensor(es[:t][-(i + 1)], n).T)
    E = np.sum(As, axis=0)

    beta[i] =np.mean(E /(2 * ( begin_idx -T_hat )))
    return beta

def update_gamma(self,es, alpha, beta,gamma, thet, unfold_cores, i, n,s):
    T_hat=len(unfold_cores)
    begin_idx=s*self._P+s*self._Q
    As= []
    for t in range(begin_idx,T_hat):
        a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p) ], axis=0)
        b = np.sum(
            [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q) ],axis=0)
        c = np.sum([gamma[ii] * unfold_cores[:t][-(s*ii + 1)] for ii in range(self._P) if i != ii], axis=0)
        d = np.sum(
            [thet[j] * self._get_unfold_tensor(es[:t][-(s*j + 1)], n) for j in range(self._Q)],axis=0)
        As.append((unfold_cores[t] - a + b-c+d) * unfold_cores[:t][-(s*i + 1)].T)
    E = np.sum(As, axis=0)

    gamma[i] =np.mean(E /(2 * (T_hat - begin_idx)))
    return gamma



def update_thet(self,es, alpha, beta,gamma, thet, unfold_cores, i, n,s):
    T_hat=len(unfold_cores)
    begin_idx=s*self._P+s*self._Q
    As= []
    for t in range(begin_idx,T_hat):
        a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p) ], axis=0)
        b = np.sum(
            [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q) ],axis=0)
        c = np.sum([gamma[ii] * unfold_cores[:t][-(s*ii + 1)] for ii in range(self._P) ], axis=0)
        d = np.sum(
            [thet[j] * self._get_unfold_tensor(es[:t][-(s*   j + 1)], n) for j in range(self._Q) if i != j], axis=0)
        As.append((unfold_cores[t] - a + b-c+d) * self._get_unfold_tensor(es[:t][-(s*i + 1)], n).T)
    E = np.sum(As, axis=0)

    thet[i] =np.mean(E /(2 * ( begin_idx -T_hat )))
    return thet



def update_Es(self, es, alpha, beta,gamma, thet, unfold_cores, i, n,s):

    T_hat = len(unfold_cores)
    begin_idx = self._p + self._q

    As = []
    for t in range(begin_idx, T_hat):
        # t-i
        a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p)], axis=0)
        b = np.sum(
            [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q) if i != j],
            axis=0)
        c = np.sum([gamma[ii] * unfold_cores[:t][-(s * ii + 1)] for ii in range(self._P)], axis=0)

        d = np.sum(
            [thet[j] * self._get_unfold_tensor(es[:t][-(s * j + 1)], n) for j in range(self._Q)], axis=0)
        As.append((unfold_cores[t] - a + b - c + d))
    E = np.sum(As, axis=0)
    for t in range(len(es)):
        es[t][i] = self._get_fold_tensor(E / (2 * (begin_idx - T_hat) * beta[i]), n, es[t][i].shape)
    return es

def _update_cores(self, n, Us, Xs, es, cores, alpha, beta,gamma, thet, lam=1):

    begin_idx = self._p + self._q
    T_hat = len(Xs)
    unfold_cores = self._get_unfold_tensor(cores, n)
    H = self._get_H(Us, n)  #U(-m).T
    for t in range(begin_idx, T_hat):
        unfold_Xs = self._get_unfold_tensor(Xs[t], n)
        a = np.sum([alpha[i] * self._get_unfold_tensor(cores[t - (i + 1)], n) for i in range(self._p)], axis=0)
        b = np.sum([ beta[i] * self._get_unfold_tensor(es[t-begin_idx][-(i+1)], n) for i in range(self._q)], axis=0) #t-begin_index  ?? t-(i+1)
        c = np.sum([gamma[ii] * unfold_cores[:t][-(s * ii + 1)] for ii in range(self._P)], axis=0)
        d = np.sum(
            [thet[j] * self._get_unfold_tensor(es[:t][-(s * j + 1)], n) for j in range(self._Q)], axis=0)

        unfold_cores[t] = 1/(1+lam) * (lam * np.dot( np.dot(Us[n].T, unfold_Xs), H.T) + a - b+c-d)
    return unfold_cores


s=3
begin,diff_data=diff_season(b,s)
data=inv_diff_season(begin,diff_data,s)

sss=mape(data,data)



print()




