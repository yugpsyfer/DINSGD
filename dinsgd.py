import torch
import numpy as np
import torch.linalg as linalg
import torch.distributed as dist
from torch.distributed import ReduceOp


class DINSGD():
    def __init__(self, params, p=1, q=10, beta=0.9, rho=1, compression_method="quantize"):
        """
        Default values have been taken from the paper https://arxiv.org/pdf/2002.04130.pdf
        Rho is number of workers - Total number of GPUs
        compression_method: ['quantize', 'one_bit', 'sparse_top_k', 'sparse_randomized']
        Compression methods have been taken from below gihub repo.
        - https://github.com/scottjiao/Gradient-Compression-Methods/blob/master/utils.py 
        """
        self.param_groups = []

        if compression_method == "quantize":
            self.compress = self.quantize
        elif compression_method == "one_bit":
            self.compress = self.one_bit
        elif compression_method == "sparse_top_k":
            self.compress = self.sparse_top_k
        elif compression_method == "sparse_randomized":
            self.compress = self.sparse_randomized
        else:
            raise('COMPRESSION METHOD NOT DEFINED')

        defaults = {"P":p, "Q":q, "Beta":beta, "Rho":rho}
        defaults.update({"params":list(params), "memory": [], "M":[]})
        
        self.param_groups.append(defaults)

        for g in self.param_groups:
            for pars in g['params']:
                g['memory'].append(torch.zeros_like(pars.data, device=torch.device('cuda')))
                g['M'].append(torch.zeros_like(pars.data, device=torch.device('cuda')))
                

    def step(self):
        rand_ = self.randomize()

        for groups in self.param_groups:
            P = groups['P']
            Q = groups['Q']
            Beta = groups['Beta']
            Rho = groups['Rho']

            for par in range(len(groups['params'])):
                
                grad = groups['params'][par].grad.data  #worker
                m_prev =  groups['M'][par]

                corrected_grad, groups['memory'][par] = self.correct_gradients(grad, groups['memory'][par]) #worker
                
                dist.all_reduce(corrected_grad, op=ReduceOp.SUM)

                groups['M'][par] = Beta * m_prev  + (1-Beta)*corrected_grad/Rho
                eta = 1/(P*linalg.norm(groups['M'][par]) + Q)
                v_t = (groups['params'][par].data - eta*groups['M'][par])*(1-rand_)
                
                groups['params'][par].data.multiply_(rand_)
                groups['params'][par].data.add_(v_t, alpha=1)


    def randomize(self): #worker
        return torch.rand(1, device=torch.device('cuda'))
    
    
    def correct_gradients(self, x, mem): #worker
        
        corrected_gradient = self.compress(mem + x)
        mem = mem + x - corrected_gradient

        return corrected_gradient, mem

    def one_bit(self, x):  #worker
        
        x_norm=torch.norm(x,p=float('inf'))
        sgn_x=((x>0).float()-0.5)*2
        compressed_x=x_norm*sgn_x
        
        return compressed_x 

    def quantize(self,x,input_compress_settings={}):  #BEST
        compress_settings={'n':6}
        # compress_settings.update(input_compress_settings)
        #assume that x is a torch tensor
        
        n=compress_settings['n']
        #print('n:{}'.format(n))
        x=x.float()
        x_norm=torch.norm(x,p=float('inf'))
        
        sgn_x=((x>0).float()-0.5)*2
        
        p=torch.div(torch.abs(x),x_norm)
        renormalize_p=torch.mul(p,n)
        floor_p=torch.floor(renormalize_p)
        compare=torch.rand_like(floor_p)
        final_p=renormalize_p-floor_p
        margin=(compare < final_p).float()
        xi=(floor_p+margin)/n
        
        Tilde_x=x_norm*sgn_x*xi
        
        return Tilde_x

    def sparse_top_k(self, x,input_compress_settings={}):
        compress_settings={'k':1/16}
        compress_settings.update(input_compress_settings)
        k=compress_settings['k']
        vec_x=x.flatten()
        d = int(len(vec_x))
        #print(d)
        k =int(np.ceil(d*k))
        #print(k)
        indices = torch.abs(vec_x).topk(k)[1]
        out_x = torch.zeros_like(vec_x)
        out_x[indices] = vec_x[indices]
        out_x=out_x.reshape(x.shape)
        #print(x.shape)
        return out_x


"""THROWS NAN"""

    # def sparse_randomized(self, x,input_compress_settings={}):      #Avoid this one
    #     max_iteration=10000
    #     compress_settings={'p':0.8}
    #     compress_settings.update(input_compress_settings)
    #     #p=compress_settings['p']
    #     #vec_x=x.flatten()
    #     #out=torch.dropout(vec_x,1-p,train=True)
    #     #out=out/p
    #     vec_x=x.flatten()
    #     d = int(len(vec_x))
    #     p=compress_settings['p']
        
    #     abs_x=torch.abs(vec_x)
    #     #d=torch.prod(torch.Tensor(x.size()))
    #     out=torch.min(p*d*abs_x/torch.sum(abs_x),torch.ones_like(abs_x))
    #     i=0
    #     while True:
    #         i+=1
    #         #print(i)
    #         if i>=max_iteration:
    #             raise ValueError('Too much operations!')
    #         temp=out.detach()
                
    #         cI=1-torch.eq(out,1).float()
    #         c=(p*d-d+torch.sum(cI))/torch.sum(out*cI)
    #         if c<=1:
    #             break
    #         out=torch.min(c*out,torch.ones_like(out))
    #         if torch.sum(1-torch.eq(out,temp)):
    #             break
        
    #     z=torch.rand_like(out)
    #     out=vec_x*(z<out).float()/out

    #     out=out.reshape(x.shape)

    #     #out=out.reshape(x.shape)
    #     return out
