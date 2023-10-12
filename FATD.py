import copy
import tensorly as tl
import scipy as sp
import numpy as np
import random
from tensorly.decomposition import tucker
from scipy.linalg import svd
from .util.MDT import MDTWrapper
from .util.functions import fit_ar_ma, svd_init


class FATD(object):
    
    def __init__(self,ts, p, d, q,s,P,Q, Rs, K, tol, seed=None, Ms_mode=4, \
        verbose=0, convergence_loss=False):
        """store all parameters in the class and do checking on taus"""
        
        self._ts = ts
        self._ts_ori_shape = ts.shape
        self._N = len(ts.shape) - 1
        self.T = ts.shape[-1]
        self._p = p
        self._d = d
        self._q = q
        self._P = P
        self._Q = Q
        self._s = s
        # self._taus = taus
        self._Rs = Rs
        self._K = K
        self._tol = tol
        self._Ms_mode = Ms_mode
        self._verbose = verbose
        self._convergence_loss = convergence_loss
        
        if seed is not None:
            np.random.seed()
        
  
    
    def _initilizer(self, T_hat, Js, Rs, Xs):
        
        # initilize Ms
        M = [ np.random.random([j,r]) for j,r in zip( list(Js), Rs )]

        # initilize es

        # es = [ [ np.random.random(Rs) for _ in range(self._q)] for t in range(begin_idx, T_hat)]

        es = [np.zeros(Rs) for t in range( T_hat)]
        alpha=[  random.random()   for i in range(self._p)]
        beta=[  random.random()   for i in range(self._q)]
        gamma=[  random.random()   for i in range(self._P)]
        thet=[  random.random()   for i in range(self._Q)]
        # gamma[0]=0.001
        # alpha[0]=0.001
        return M, es,alpha, beta, gamma, thet

    def _test_initilizer(self, trans_data, Rs):
        
        T_hat = trans_data.shape[-1]
        # initilize Ms
        M = [ np.random.random([j,r]) for j,r in zip( list(trans_data.shape)[:-1], Rs )]

        # initilize es
        begin_idx = s*self._P + s*self._Q  
        # es = [ [ np.zeros(Rs) for _ in range(self._q)] for t in range(begin_idx, T_hat)]

        es = [np.zeros(Rs) for t in range(T_hat)]
        return M, es

    
    def _initilize_M(self, T_hat, Xs, Rs):

        haveNan = True
        while haveNan:
            factors = svd_init(Xs[0], range(len(Xs[0].shape)), ranks=Rs)
            haveNan = np.any(np.isnan(factors))
        return factors  

    # def _inverse_MDT(self, mdt, data, taus, shape):
    #     return mdt.inverse(data, taus, shape)
    
    def _get_cores(self, Xs, Ms):
        s=[u.T for u in Ms]
        cores = [ tl.tenalg.multi_mode_dot( x, [u.T for u in Ms], modes=[i for i in range(len(Ms))] ) for x in Xs]
        return cores
    
    def _estimate_ar_ma(self, cores, p, q):
        cores = copy.deepcopy(cores)
        alpha, beta = fit_ar_ma(cores, p, q)

        return alpha, beta

    def _estimate_s_ar_ma(self,es, alpha, beta,gamma, thet, unfold_cores, n):
        s=self._s
        for i in range(self._p):
            alpha = self.update_alpha(es, alpha, beta, gamma, thet, unfold_cores, i, n, s)
        # for i in range(self._q):
        #     beta = self.update_beta(es, alpha, beta, gamma, thet, unfold_cores, i, n, s)
        for i in range(self._P):
            gamma = self.update_gamma(es, alpha, beta, gamma, thet, unfold_cores, i, n, s)
        # for i in range(self._Q):
        #     thet = self.update_thet(es, alpha, beta, gamma, thet, unfold_cores, i, n, s)

        return alpha, beta, gamma, thet

    def _get_fold_tensor(self, tensor, mode, shape):
        if isinstance(tensor,list):
            return [ tl.base.fold(ten, mode, shape) for ten in tensor ]
        elif isinstance(tensor, np.ndarray):
            return tl.base.fold(tensor, mode, shape)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

    def _get_unfold_tensor(self, tensor, mode):   #张量展开
        
        if isinstance(tensor, list):
            return [ tl.base.unfold(ten, mode) for ten in tensor]
        elif isinstance(tensor, np.ndarray):
            return tl.base.unfold(tensor, mode)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")   

    def _update_Ms(self, Ms, Xs, unfold_cores, n):

        T_hat = len(Xs)
        M = len(Ms)
        begin_idx = self._p + self._q

        # s=self._s
        # begin_idx = s*self._P + s*self._Q


        H = self._get_H(Ms, n)
        # orth in J3
        if self._Ms_mode == 1:
            if n<M-1:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Ms[n] = temp / np.linalg.norm(temp)
            else:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                M_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Ms[n] = np.dot(M_, V_)
        # orth in J1 J2
        elif self._Ms_mode == 2:
            if n<M-1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                M_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Ms[n] = np.dot(M_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Ms[n] = temp / np.linalg.norm(temp)
        # no orth      
        elif self._Ms_mode == 3:
            As = []
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            a = sp.linalg.pinv(np.sum(As, axis=0))
            b = np.sum(Bs, axis=0)
            temp = np.dot(a, b)
            Ms[n] = temp / np.linalg.norm(temp)
        # all orth   
        elif self._Ms_mode == 4:
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            b = np.sum(Bs, axis=0)
            # b = b.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
            b=np.array(b)
            M_, _, V_ = svd(b, full_matrices=False)
            Ms[n] = np.dot(M_, V_)
        # only orth in J1

        elif self._Ms_mode == 5:
            if n==0:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                M_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Ms[n] = np.dot(M_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Ms[n] = temp / np.linalg.norm(temp)
        # only orth in J2
        elif self._Ms_mode == 6:
            if n==1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                M_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Ms[n] = np.dot(M_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Ms[n] = temp / np.linalg.norm(temp)
        return Ms

    def _update_Es(self, es, alpha, beta, unfold_cores, i, n):

        T_hat = len(unfold_cores)
        begin_idx = self._p + self._q

        As = []
        for t in range(begin_idx, T_hat):
                                                  #t-i
            a = np.sum([alpha[ii] * unfold_cores[:t][-(ii+1)] for ii in range(self._p)] , axis=0)
            b = np.sum([beta[j] * self._get_unfold_tensor(es[:t][-(j+1)], n)  for j in range(self._q) if i!=j ] , axis=0)
            As.append(unfold_cores[t] - a + b)
        E = np.sum(As, axis=0)

        for t in range(begin_idx,len(es)):
            es[:t][-(i+1)] = self._get_fold_tensor(E / (2*(begin_idx - T_hat) * beta[i]), n, es[:t][-(i+1)].shape)
        return es

##########################################################################

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
            es[-(t+1)] = self._get_fold_tensor(E / (2 * (begin_idx - T_hat) * beta[i]), n, es[-(t+1)].shape)

        return es

    def update_alpha(self, es, alpha, beta, gamma, thet, unfold_cores, i, n, s):
        T_hat = len(unfold_cores)
        begin_idx = s * self._P + s * self._Q
        As = []
        for t in range(begin_idx, T_hat):
            a = np.sum(
                [alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p) if i != ii], axis=0)
            b = np.sum(
                [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q)], axis=0)
            c = np.sum(
                [gamma[ii] * unfold_cores[:t][-(s * ii + 1)] for ii in range(self._P)], axis=0)
            d = np.sum(
                [thet[j] * self._get_unfold_tensor(es[:t][-(s * j + 1)], n) for j in range(self._Q)], axis=0)
            As.append((unfold_cores[t] - a + b - c + d) @ np.linalg.pinv(unfold_cores[:t][-(i + 1)])  )
        E = np.sum(As, axis=0)

        alpha[i] = np.mean(E / (100 * (T_hat - begin_idx)))
        return alpha

    def update_beta(self, es, alpha, beta, gamma, thet, unfold_cores, i, n, s):
        T_hat = len(unfold_cores)
        begin_idx = s * self._P + s * self._Q
        As = []
        for t in range(begin_idx, T_hat):
            a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p)], axis=0)
            b = np.sum(
                [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q) if i != j], axis=0)
            c = np.sum([gamma[ii] * unfold_cores[:t][-(s * ii + 1)] for ii in range(self._P)], axis=0)
            d = np.sum(
                [thet[j] * self._get_unfold_tensor(es[:t][-(s * j + 1)], n) for j in range(self._Q)], axis=0)
            # aaa=(unfold_cores[t] - a + b - c + d)
            bbbb=self._get_unfold_tensor(es[:t][-(i + 1)], n).T

            As.append((unfold_cores[t] - a + b - c + d) @ self._get_unfold_tensor(es[:t][-(i + 1)], n).T  )
        E = np.sum(As, axis=0)

        beta[i] = np.mean(E / (100 * (begin_idx - T_hat)))
        return beta

    def update_gamma(self, es, alpha, beta, gamma, thet, unfold_cores, i, n, s):
        T_hat = len(unfold_cores)
        begin_idx = s * self._P + s * self._Q
        As = []
        for t in range(begin_idx, T_hat):
            a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p)], axis=0)
            b = np.sum(
                [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q)], axis=0)
            c = np.sum([gamma[ii] * unfold_cores[:t][-(s * ii + 1)] for ii in range(self._P) if i != ii], axis=0)
            d = np.sum(
                [thet[j] * self._get_unfold_tensor(es[:t][-(s * j + 1)], n) for j in range(self._Q)], axis=0)
            As.append((unfold_cores[t] - a + b - c + d) @   np.linalg.pinv(unfold_cores[:t][-(s * i + 1)]) )
        E = np.sum(As, axis=0)

        gamma[i] = np.mean(E / (100 * (T_hat - begin_idx)))
        return gamma

    def update_thet(self, es, alpha, beta, gamma, thet, unfold_cores, i, n, s):
        T_hat = len(unfold_cores)
        begin_idx = s * self._P + s * self._Q
        As = []
        for t in range(begin_idx, T_hat):
            a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p)], axis=0)
            b = np.sum(
                [beta[j] * self._get_unfold_tensor(es[:t][-(j + 1)], n) for j in range(self._q)], axis=0)
            c = np.sum([gamma[ii] * unfold_cores[:t][-(s * ii + 1)] for ii in range(self._P)], axis=0)
            d = np.sum(
                [thet[j] * self._get_unfold_tensor(es[:t][-(s * j + 1)], n) for j in range(self._Q) if i != j], axis=0)
            As.append((unfold_cores[t] - a + b - c + d) @ self._get_unfold_tensor(es[:t][-(s * i + 1)], n).T)
        E = np.sum(As, axis=0)

        thet[i] = np.mean(E / (2 * (begin_idx - T_hat)))
        return thet

    def update_Es(self, es, alpha, beta, gamma, thet, unfold_cores, i, n):
        s=self._s
        T_hat = len(unfold_cores)
        begin_idx = s*self._P + s*self._Q

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
        for t in range(begin_idx,len(es)):
            es[-(t+1)] =  self._get_fold_tensor(E / (2 * (begin_idx - T_hat) * beta[i]), n, es[:t][-i].shape)

        return es

    def _update_cores(self, n, Ms, Xs, es, cores, alpha, beta, lam=1):
        s=self._s
        begin_idx = self._p + self._q
        T_hat = len(Xs)
        unfold_cores = self._get_unfold_tensor(cores, n)
        H = self._get_H(Ms, n)  # M(-m).T
        for t in range(begin_idx, T_hat):
            unfold_Xs = self._get_unfold_tensor(Xs[t], n)
            a = np.sum([alpha[i] * self._get_unfold_tensor(cores[t - (i + 1)], n) for i in range(self._p)], axis=0)

            b = np.sum([beta[i] * self._get_unfold_tensor(es[:t][-(i + 1)], n) for i in range(self._q)],
                       axis=0)  #

            unfold_cores[t] = 1 / (1 + lam) * (lam * np.dot(np.dot(Ms[n].T, unfold_Xs), H.T) + a - b )
        return unfold_cores


    def update_cores(self, n, Ms, Xs, es, cores, alpha, beta, gamma, thet, lam=1):
        s=self._s
        begin_idx = s*self._P + s*self._Q
        T_hat = len(Xs)
        unfold_cores = self._get_unfold_tensor(cores, n)
        H = self._get_H(Ms, n)  # M(-m).T
        for t in range(begin_idx, T_hat):
            unfold_Xs = self._get_unfold_tensor(Xs[t], n)
            a = np.sum([alpha[i] * self._get_unfold_tensor(cores[t - (i + 1)], n) for i in range(self._p)], axis=0)

            b = np.sum([beta[i] * self._get_unfold_tensor(es[:t][-(i + 1)], n) for i in range(self._q)],
                       axis=0)  #
            c = np.sum([gamma[ii] * self._get_unfold_tensor(cores[t - (s*ii + 1)], n) for ii in range(self._P)], axis=0)
            d = np.sum(
                [thet[j] * self._get_unfold_tensor(es[:t][-(s * j + 1)], n) for j in range(self._Q)], axis=0)
            unfold_cores[t] = 1 / (1 + lam) * (lam * np.dot(np.dot(Ms[n].T, unfold_Xs), H.T) + a - b + c - d)
        return unfold_cores


    def _compute_convergence(self, new_M, old_M):
        
        new_old = [ n-o for n, o in zip(new_M, old_M)]
        
        a = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_old], axis=0)
        b = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_M], axis=0)
        return a/b
    
    def _tensor_difference(self, d, tensors, axis):
        """
        get d-order difference series
        Arg:
            d: int, order
            tensors: list of ndarray, tensor to be difference
        Return:
            begin_tensors: list, the first d elements, used for recovering original tensors
            d_tensors: ndarray, tensors after difference
        """
        d_tensors = tensors
        begin_tensors = []

        for _ in range(d):
            begin_tensors.append(d_tensors[0])
            d_tensors = list(np.diff(d_tensors, axis=axis))
        
        return begin_tensors, d_tensors

    def diff_season(self,data, s):
        begin = data[0:s]
        diff_data = []
        i = len(data) - 1
        while (i - s >= 0):
            diff_data.insert(0,data[i]-data[i-s])
            i -= 1
        return begin, diff_data

    def inv_diff_season(self,begin, diff_data, s):
        data = begin
        max_idx = len(begin) + len(diff_data)
        r = 0
        i = 0
        while i < len(diff_data):
            for j in range(s):
                data.append(data[s * r + j] + diff_data[i])
                i += 1
                if i >= len(diff_data): break
            r += 1
        return data

    def _tensor_reverse_diff(self, d, begin, tensors, axis):
        """
        recover original tensors from d-order difference tensors
        
        Arg:
            d: int, order
            begin: list, the first d elements
            tensors: list of ndarray, tensors after difference
        
        Return:
            re_tensors: ndarray, original tensors
        
        """
        
        re_tensors = tensors      
        for i in range(1, d+1):
            re_tensors = list(np.cumsum(np.insert(re_tensors, 0, begin[-i], axis=axis), axis=axis))
         
        return re_tensors
    
    # def _update_cores(self, n, Ms, Xs, es, cores, alpha, beta, lam=1):
    #
    #     begin_idx = self._p + self._q
    #     T_hat = len(Xs)
    #     unfold_cores = self._get_unfold_tensor(cores, n)
    #     H = self._get_H(Ms, n)  #M(-m).T
    #     for t in range(begin_idx, T_hat):
    #         unfold_Xs = self._get_unfold_tensor(Xs[t], n)
    #         b = np.sum([ beta[i] * self._get_unfold_tensor(es[:t][-(i+1)], n) for i in range(self._q)], axis=0) #t-begin_index  ?? t-(i+1)
    #         a = np.sum([ alpha[i] * self._get_unfold_tensor(cores[t-(i+1)], n) for i in range(self._p)], axis=0 )
    #         unfold_cores[t] = 1/(1+lam) * (lam * np.dot( np.dot(Ms[n].T, unfold_Xs), H.T) + a - b)
    #     return unfold_cores

    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs

    def _get_H(self, Ms, n): #M(-m).T
        ab=Ms[::-1]
        Hs = tl.tenalg.kronecker([u.T for u, i in zip(Ms[::-1], reversed(range(len(Ms)))) if i!= n ])
        return Hs
    
    def run(self):
        """run the program

        Returns
        -------
        result : np.ndarray, shape (num of items, num of time step +1)
            prediction result, included the original series

        loss : list[float] if self.convergence_loss == True else None
            Convergence loss

        """

        
        result, loss = self._run()
        
        if self._convergence_loss:
            
            return result, loss            
        
        return result, None

    def _run(self):

        Xs = self._get_Xs(self._ts)

        # begin_s,Xs = self.diff_season(Xs,self._s)
        if self._d!=0:
            begin, Xs = self._tensor_difference(self._d, Xs, 0)

        # for plotting the convergence loss figure
        con_loss = []

        # Step 2
        # initialize Ms
        Ms, es,alpha, beta, gamma, thet = self._initilizer(len(Xs), Xs[0].shape, self._Rs, Xs)

        for k in range(self._K):

            old_Ms = Ms.copy()
            # get cores
            cores = self._get_cores(Xs, Ms)

            # # estimate the coefficients of AR and MA model
            # alpha, beta = self._estimate_ar_ma(cores, self._p, self._q)
            # for n in range(len(self._Rs)):    #mode n
            #
            #     cores_shape = cores[0].shape
            #     unfold_cores = self._update_cores(n, Ms, Xs, es, cores, alpha, beta , lam=1)
            #     cores = self._get_fold_tensor(unfold_cores, n, cores_shape)
            #     # update Ms
            #     Ms = self._update_Ms(Ms, Xs, unfold_cores, n)
            #     for i in range(self._q):
            #         # update Es
            #         es = self._update_Es(es, alpha, beta,   unfold_cores, i, n)
            # alpha, beta = self._estimate_ar_ma(cores, self._p, self._q)
            for n in range(len(self._Rs)):  # mode n

                cores_shape = cores[0].shape
                unfold_cores = self.update_cores(n, Ms, Xs, es, cores, alpha, beta, gamma, thet, lam=1)
                cores = self._get_fold_tensor(unfold_cores, n, cores_shape)
                # update Ms

                Ms = self._update_Ms(Ms, Xs, unfold_cores, n)

                for i in range(self._q):
                    # update Es
                    es = self.update_Es(es, alpha, beta, gamma, thet, unfold_cores, i, n)
                alpha, beta, gamma, thet = self._estimate_s_ar_ma(es, alpha, beta, gamma, thet, unfold_cores, n)
                # print(f"{alpha}, {beta}, {gamma}, {thet}\n" )


            # convergence check:
            convergence = self._compute_convergence(Ms, old_Ms)
            con_loss.append(convergence)

            if k%2 == 0:
                if self._verbose == 1:  #verbose 是否展示迭代信息  默认值为0
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                    #print("alpha: {}, beta: {}".format(alpha, beta))

            if self._tol > convergence:
                if self._verbose == 1:
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                    print("alpha: {}, beta: {}".format(alpha, beta))
                break


        # Step 3: Forecasting
        #get cores
        cores = self._get_cores(Xs, Ms)


        alpha, beta = self._estimate_ar_ma(cores, self._p, self._q)
                                                                      #[::-1]列表逆置
        # for n in range(len(self._Rs)):
        #     alpha, beta, gamma, thet = self._estimate_s_ar_ma(es, alpha, beta, gamma, thet, unfold_cores, n)

        ssss=cores[-self._p:][::-1]
        new_core = np.sum([al * core for al, core in zip(alpha, cores[-self._p:][::-1])], axis=0) \
                - np.sum([be * e for be, e in zip(beta, es[-self._q:][::-1])], axis=0)

        s=self._s
        a1=[]
        for i in range(self._P):
            a1.append(cores[-s*i])
        a2 = []
        for i in range(self._Q):
            a2.append(es[-s * i])
        # new_core =new_core+  np.sum([gam * core for gam, core in zip(gamma, cores)], axis=0) \
        #            - np.sum([the * e for the, e in zip(thet, es)], axis=0)

        new_X = tl.tenalg.multi_mode_dot(new_core, Ms)
        Xs.append(new_X)

        if self._d != 0:
            Xs = self._tensor_reverse_diff(self._d, begin, Xs, 0)
        # Xs = self.inv_diff_season(begin_s, Xs, self._s)
        mdt_result = Xs[-1]
        
        return mdt_result ,con_loss
