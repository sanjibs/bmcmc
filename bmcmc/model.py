#Copyright (c) 2016 Sanjib Sharma
"""
A module for performing Markov chain Monte Carlo sampling 
using adaptive proposal distribution. It also allows 
sampling of a hierarchical bayesian model using Metropolis 
within gibbs scheme. To use this one has to define a model 
by creating a subclass out of base class bmcmc.Model. 
The subclass should define three methods as shown below. 
Then calling the method sample() invokes the MCMC sampler.  


Example:

class gauss1(bmcmc.Model):
    def set_descr(self):
        # setup descriptor
        # self.descr[varname]=[type,value,sigma,latexname,min_val,max_val]
        self.descr['mu']     =['p',0.0,1.0,r'$\mu$ ',-500,500.0]
        self.descr['sigma']  =['p',1.0,1.0,r'$\sigma$ ',1e-10,1e3]
        self.descr['x']      =['d',0.0,1.0,r'$x$ ',-500.0,500.0]
        self.descr['xt']     =['d',0.0,1.0,r'$x_t$ ',-500.0,500.0]
        self.descr['sigma_x']=['p',0.5,1.0,r'$\sigma_x$ ',1e-10,1e3]

    def set_args(self):
        # setup data points 
        np.random.seed(11)
        self.args['x']=np.random.normal(self.descr['mu'][1],self.descr['sigma'][1],self.eargs['dsize'])
        self.args['x']=np.random.normal(self.args['x'],self.args['sigma_x'])
    

    def lnfunc(self,args):
        # setup the lnprob function
        temp1=-np.square(args['xt']-args['mu'])/(2*args['sigma']*args['sigma'])-np.log(np.sqrt(2*np.pi)*args['sigma'])
        temp=temp1-np.square(args['x']-args['xt'])/(2*args['sigma_x']*args['sigma_x'])-np.log(np.sqrt(2*np.pi)*args['sigma_x'])
        return temp

model=gauss1(eargs={'dsize':100})
model.sample(['xt','mu','sigma'],10000)
model.info()

"""

import matplotlib.pyplot  as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import ast
#import autil
#import sutil
import ebf
#import acor

import numpy as np
import scipy as sp


def autocorr(x):
    x=x-np.mean(x)
    y=np.correlate(x,x,mode='full')
    return y[y.size/2:]/y[y.size/2]

def sround(x,sig1,sig2=None):    
    if sig2 == None:
        sig=sig1
    else:
        sig=(sig2-sig1)/2.0
    d2=np.floor(np.log10(sig))
    return '%g +/- %g'%(np.round(x*(10**(-d2)))/10.0**(-d2),np.round(sig*(10**(-d2)))/10.0**(-d2))

def stat_text(x,uplow=False):
    temp=np.percentile(x,[16.0,84.0,50.0])
    sig=(temp[1]-temp[0])/2.0
    xmean=np.mean(x)
#    sig=np.std(x)
    d2=np.floor(np.log10(sig))
    if uplow:
        return r'$%g^{+%g}_{-%g}$'%(np.round(xmean*(10**(-d2)))/10.0**(-d2),np.round((xmean-temp[0])*(10**(-d2)))/10.0**(-d2),np.round((temp[1]-xmean)*(10**(-d2)))/10.0**(-d2))
    else:
        return r'$%g \pm %g$'%(np.round(xmean*(10**(-d2)))/10.0**(-d2),np.round(sig*(10**(-d2)))/10.0**(-d2))



class Model():
    """
    The base Model class from which the user should create a subclass 
    and write the following three methods.

    class mymodel(bmcmc.Model):
          def set_descr(self):
              # should define a dicts self.descr 
              self.descr[varname]=[type,value,sigma,latexname,min_val,max_val]
              ....................
              ....................

          def set_args(self):
              # should define a dict self.args specifying the data points
              # Variables that will be used by self.lnfunc but are not 
              # defined in self.descr must be specified here.
              # Variables specified in self.descr will be automatically 
              # assigned to self.args via
              #  self.args[argname]=self.descr['varname'][1]
              # and will be available here.  
              # However, you can always redefine them here or by 
              # interacting with a created object of the model. 
              self.args[argname]=
              ....................
              ....................

          def lnfunc(self,args):
              # define the logarithm of the function you want to sample
              # using only args
              ....................
              ....................
              return temp

    Information about self.desrc-
    varname  : string, name of variable
    type     : string, 'p' (for parameter) or 'd' (for data)  
    value    : float, starting value of the variable
    sigma    : float, starting value of stddev dev for the variable to be sued in proposal dist
    latexname: string, latex name used in printing, e.g., '$\alpha$'
    min_val  : float, minimum value of the variable, defines the range that MCMC will explore
    max_val  : float, maximum value of the variable, defines the range that MCMC will explore

    """
    def __init__(self,eargs=None):
        if type(eargs) is int:
            eargs={'dsize':eargs}
        self.eargs=eargs        
        self.args={}
        self.descr={}
        self.set_descr()
        self.lnprob_min=-1e38
        self.stoch=False
        for name in self.descr.keys():
            if (self.descr[name][0]=='l0')or(self.descr[name][0]=='p'):
                self.args[name]=np.float64(self.descr[name][1])
        self.set_args()
        for name in self.descr.keys():
            if (self.descr[name][0]=='l1')or(self.descr[name][0]=='d'):
                if name not in self.args.keys(): 
                    self.args[name]=np.zeros(self.eargs['dsize'],dtype=np.float64)+self.descr[name][1]

    def lngauss(self,x,sigma):
        """
        logarithm of normal distribution with zero mean and stdddev sigma
        Can be used in lnfunc
        """
        temp=np.sqrt(2*np.pi)
        return -0.5*np.square(x/sigma)-np.log(temp*sigma)

    def lnprob(self,varnames,y,gibbs=False):
        """
        For use by MCMC sampler
        """
        if len(varnames)>0:
            if gibbs:
                self.mask=np.zeros(self.eargs['dsize'],dtype=np.int64)
                for i,name in enumerate(varnames):
                    self.args[name]=y[i]
                    ind=np.where((y[i]<self.descr[name][4])|(y[i]>self.descr[name][5]))[0]
                    self.mask[ind]=1
            else:
                self.mask=0
                for i,name in enumerate(varnames):
                    self.args[name]=y[i]
                    if y[i]<self.descr[name][4]:
                        self.mask=1
                    if y[i]>self.descr[name][5]:
                        self.mask=1

        res=self.lnfunc(self.args)
        if gibbs:
            ind=np.where(self.mask==1)[0]
            res[ind]=self.lnprob_min            
        else:
            if self.mask==1:
                if hasattr(res,'len'):
                    res[:]=self.lnprob_min
                else:
                    res=self.lnprob_min
        return res
        # self.set_args(varnames,y,gibbs)
        # return self.get_res(self.lnfunc(self.args),gibbs)


    def read_args(self,filename,varnames):
        """
        Read arguments from an ebf file
        Can be used in set_args()
        """
        for name in varnames:
            self.args[name]=ebf.read(filename,'/'+name)

    def get_guess_variance(self,names):
        """
        For use by MCMC sampler
        """
        return np.array([self.descr[name][2]*self.descr[name][2] for name in names])

    def get_guess_values(self,names):
        """
        For use by MCMC sampler
        """
        return np.array([self.descr[name][1] for name in names])


    def sample(self,varnames,iterations,atime=10,ptime=1000):
        """
        Run a markov chain monte sampler
        Parameters
        ----------
        varnames: a list of names specifying variables over which to run MCMC
        iterations: desired length of the MCMC chain 
        atime: iterations after which to adapt the proposal distribution        
        ptime: iterations after which to print summary on screen       
        Creates
        --------
        names0: the name of parameters of the model that are varied (top level)
        names1: the name of parameters of the model that are varied (next level)
        mu:     the mean value of all estimated parameters
        sigma:  the stddev of all estimated parameters
        chain:  mcmc chain for each estimated parameter at the top level
        """
        pnames=[]
        dnames=[]
        for name in varnames:
            if (self.descr[name][0] == 'l0')or(self.descr[name][0] == 'p'):
                pnames.append(name)
            else:
                dnames.append(name)
        if len(dnames)>0:
            s1=_MWG(self,dnames,atime,ptime)
            if len(pnames)>0:
                s2=_MH(self,pnames,atime,ptime)
                for i in range(iterations):
                    s1.next()
                    s2.next()
            else:
                s2=None
                for i in range(iterations):
                    s1.next()
        else:
            s1=None
            if len(pnames)>0:
                s2=_MH(self,pnames,atime,ptime)
                for i in range(iterations):
                    s2.next()
            else:
                s2=None

        if s1 is not None:
            self.mu,self.sigma=s1.get_musigma()
        else:
            self.mu={}
            self.sigma={}

        if s2 is not None:
            self.chain=s2.get_chain()
            for name in self.chain.keys():
                self.mu[name]=np.mean(self.chain[name])
                self.sigma[name]=np.std(self.chain[name])

        self.names0=pnames
        self.names1=dnames

    def best_fit(self,burn=1000):
        return np.array([np.mean(self.chain[key][burn:]) for key in self.names0])
        

    def info(self,burn=1000,latex=False):
        """
        Print statistics of mcmc chain
        """
        if latex:
            s='\\begin{tabular} { l  c} \n'
            s=s+'Parameter &  16 and 84 \% limits\\\ \n'
            s=s+'\hline \n'
            for key in self.names0:
                x=self.chain[key][burn:]
                s=s+self.descr[key][3]+' & '+stat_text(x,uplow=True)+' \\\ \n'
            s=s+'\hline \n'
            s=s+'\end{tabular} \n'
            print s
        else:            
            print '%4s %16s %12s %12s [%12s, %12s, %12s]'%('no','name','mean','stddev','16%','50%','84%')
            if len(self.names0)>0:
                print 'Level-0'
                for i,name in enumerate(self.names0):
                    x=self.chain[name][burn:]
                    temp=np.percentile(x,[16.0,50.0,84.0])
                    print '%4i %16s %12g %12g [%12g, %12g, %12g]'%(i,name,np.mean(x),np.std(x),temp[0],temp[1],temp[2])

            if len(self.names1)>0:
                print 'Level-1'
                for i,name in enumerate(self.names1):
                    print '%4i %16s %12g %12g'%(i,name,np.mean(self.mu[name]),np.mean(self.sigma[name]))

    def write(self,outname):
        ebf.initialize(outname)
        ebf.write(outname,'/descr',str(self.descr),'a')
        if len(self.names1) > 0:
            ebf.write(outname,'/mu/',self.mu,'a')
            ebf.write(outname,'/sigma/',self.sigma,'a')
            ebf.write(outname,'/names1',np.array(self.names1),'a')
        if len(self.names0) > 0:
            ebf.write(outname,'/chain/',self.chain,'a')
            ebf.write(outname,'/names0',np.array(self.names0),'a')

    def read(self,outname):
        if ebf.containsKey(outname,'/names0'):
            x=ebf.read(outname,'/names0')            
            self.names0=[str(temp) for temp in x]
            self.chain=ebf.read(outname,'/chain/')
        else:
            self.names0=[]
        if ebf.containsKey(outname,'/names1'):
            x=ebf.read(outname,'/names1')            
            self.names1=[str(temp) for temp in x]
            self.mu=ebf.read(outname,'/mu/')
            self.sigma=ebf.read(outname,'/sigma/')
        else:
            self.names1=[]

        self.descr=ast.literal_eval(ebf.read(outname,'/descr')[0])

    def plot(self,keys=None,burn=1000):
        if keys is None:
            keys=self.names0
        k=0
        #plm=putil.Plm1(rows=2,cols=2,xmulti=True,ymulti=True,slabel=False)
        for i in range(len(keys)):
            for j in range(len(keys)):
                k=k+1
                if i==j:
                    x=self.chain[keys[i]][burn:]
                    plt.subplot(len(keys),len(keys),k)
                    #sig=np.std(self.chain[keys[i]][burn:])
                    xmean=np.mean(x)
                    nbins=np.max([20,x.size/1000])
                    plt.hist(x,bins=nbins,normed=True,histtype='step')
                    plt.axvline(np.mean(self.chain[keys[i]][burn:]),lw=2.0,color='g')
                    if i == (len(keys)-1):
                        plt.xlabel(self.descr[keys[i]][3])
                    plt.text(0.05,0.7,stat_text(self.chain[keys[i]][burn:]),transform=plt.gca().transAxes)
                    plt.gca().xaxis.set_major_locator(MaxNLocator(3, prune="both"))
                    plt.gca().yaxis.set_major_locator(MaxNLocator(3, prune="both"))
                    plt.gca().set_yticklabels([])
                else:
                    if i > j:
                        plt.subplot(len(keys),len(keys),k)
                        x=self.chain[keys[j]][burn:]
                        y=self.chain[keys[i]][burn:]
                        nbins=np.max([32,x.size/1000])
                        plt.hist2d(x,y,bins=[nbins,nbins],norm=LogNorm())
                        plt.axvline(np.mean(self.chain[keys[j]][burn:]),lw=2.0)
                        plt.axhline(np.mean(self.chain[keys[i]][burn:]),lw=2.0)
                        if j == 0:
                            plt.ylabel(self.descr[keys[i]][3])
                        else:
                            plt.gca().set_yticklabels([])
                        if i == (len(keys)-1):
                            plt.xlabel(self.descr[keys[j]][3])
                        else:
                            plt.gca().set_xticklabels([])

                        plt.gca().xaxis.set_major_locator(MaxNLocator(3, prune="both"))
                        plt.gca().yaxis.set_major_locator(MaxNLocator(3, prune="both"))
                        #plt.colorbar(pad=0.0,fraction=0.1)
        plt.subplots_adjust(hspace=0.15,wspace=0.1)

class _MWG():
    """
    The Metropolis within Gibbs sampler with autotuning of proposal 
    distributions. In hierarchical model this samples the properties 
    of each data point rather than  the global model parameters.
    """
    def __init__(self,model,varnames,atime=10,ptime=1000):
        """
        Parameters        
        ----------
        model: a subclass of Model()
        varnames: a list of name of variables over which to do mcmc 
                  For each variable there must be a corresponding entry 
                  in Model().descr
        atime: iterations after which to adapt the proposal distribution        
        ptime: iterations after which to print summary on screen       
        """
        if type(varnames) == str:
            varnames=[varnames]

        self.model=model
        self.varnames=varnames
        vsize=len(self.varnames)
        dsize=model.eargs['dsize']
        self.variance=np.zeros((vsize,dsize),dtype=np.float64)
        self.chain=np.zeros((vsize,dsize),dtype=np.float64)
        self.chainb=[]
        for i,name in enumerate(self.varnames):
            self.variance[i,:]=self.model.get_guess_variance([name])[0]
            self.chain[i,:]=self.model.get_guess_values([name])[0]

        # self.variance=(np.zeros((dsize,vsize),dtype=np.float64)+self.model.get_guess_variance(varnames)[0]).T
        # self.chain=(np.zeros((dsize,vsize),dtype=np.float64)+self.model.get_guess_values(varnames)[0]).T

        self.alpha=np.ones(dsize,dtype=np.float64)
        self.model.lnprob_cur=self.model.lnprob(self.varnames,self.chain,gibbs=True)
        self.mu=np.zeros((vsize,dsize),dtype=np.float64)
        self.x2=np.zeros((vsize,dsize),dtype=np.float64)
        self.mu1=np.zeros((vsize,dsize),dtype=np.float64)

        self.atime=atime
        self.ptime=ptime
        self.lam=0.0
        self.delta=0.65
        self.i=0
        al=[0.0,0.44,0.352,0.316,0.279,0.275,0.266]
        self.alpha0=al[np.min([len(varnames),6])]

    def next(self):
        """
        advance mcmc to next step
        """
        y=np.random.normal(self.chain,np.sqrt(np.exp(self.lam)*self.variance))
        lnprob_old=self.model.lnprob_cur
        lnprob_new=self.model.lnprob(self.varnames,y,gibbs=True)

        self.alpha_cur=(lnprob_new-lnprob_old).clip(max=0.0)
        ind=np.where(np.log(np.random.ranf(y.shape[1])) < self.alpha_cur)[0]

        self.chain[:,ind]=y[:,ind]
        self.model.lnprob_cur[ind]=lnprob_new[ind]
        for i,name in enumerate(self.varnames):
            self.model.args[name]=self.chain[i]
        alpha=np.zeros(y.shape[1],dtype=np.float64)
        alpha[ind]=1.0

        self.i=self.i+1
        gamma=(1.0/self.i)
#        y=self.chain
        self.alpha=self.alpha*(1-gamma)+alpha*gamma
        self.mu=self.mu*(1-gamma)+self.chain*gamma
        self.x2=self.x2*(1-gamma)+(self.chain*self.chain)*gamma

        if self.atime>0:
            self._adapt()
        if (self.i)%self.ptime == 0:
            self._stats()

    def _adapt(self):
        #(i+1) needed to avoid getting stuck initially with cov=0 when alpha=0
        gamma=(1.0/(self.i+1))**self.delta
        self.lam=self.lam+(np.exp(self.alpha_cur)-self.alpha0)*gamma
        temp=self.chain-self.mu1
        self.variance=self.variance+(temp*temp-self.variance)*gamma
        self.mu1=self.mu1+(self.chain-self.mu1)*gamma
        # self.var1=self.x2*(1-gamma)+(y*y)*gamma
        # cov1=np.outer(self.chain[-1]-self.mu,self.chain[-1]-self.mu)
        # self.cov=self.cov+(cov1-self.cov)*gamma


    def get_musigma(self,burn=0):
        """
        Get the mcmc output
        Returns
        -------
        mu   : the mean value of each variable in varnames 
               and each data point
        sigma: the stddev of each variable in varnames 
               and each data point
        """
        temp=self.x2-self.mu*self.mu
        temp=np.sqrt(temp.clip(min=0.0))
        mu={}
        sigma={}
        for i,name in enumerate(self.varnames):
            mu[name]=self.mu[i]
            sigma[name]=temp[i]
        return mu,sigma

    def _stats(self):
        """
        Print the statistics
        First line: iteration, desired acceptance ratio, 
                    acceptance ratio for full sample, 
                    lambda
        Second line: name,mean,stddev               
        """
        print '%-4s %10i %6.3f %6.3f %6.3f'%('MWG',self.i,self.alpha0,np.mean(self.alpha),np.mean(self.lam))
        for i,name in enumerate(self.varnames):
            print '%16s %12g %12g'%(name, np.mean(self.mu[i]), np.mean(np.sqrt((self.x2[i]-self.mu[i]*self.mu[i]).clip(min=0.0)))) 

#        print np.sqrt(self.x2-self.mu*self.mu)
        # for i,name in enumerate(varnames):
        #     print i, name, mu[i], sig[i] 

    def info(self):
        # mu=np.mean(self.mu)
        # sig=np.sqrt(np.max([0.0,np.mean(self.x2-self.mu*self.mu)]))
        # temp=self.x2-self.mu*self.mu
        # temp=np.sqrt(temp.clip(min=0.0))
        print '%4s %16s %12s %12s'%('no','name','mean','stddev')
        for i,name in enumerate(self.varnames):
            print '%4i %16s %12g %12g'%(i, name, np.mean(self.mu[i]), np.mean(np.sqrt((self.x2[i]-self.mu[i]*self.mu[i]).clip(min=0.0))) )

    def write(self,filename,mode):
        ebf.write(filename,'/mwg/varnames',self.varnames,mode)
        ebf.write(filename,'/mwg/alpha',np.array(self.alpha),'a')
        ebf.write(filename,'/mwg/mu',np.array(self.mu),'a')
        ebf.write(filename,'/mwg/sigma',np.array(np.sqrt((self.x2-self.mu*self.mu).clip(min=0.0))),'a')

class _MH():
    """
    The Metropolis hastings sampler with autotuning of proposal distributions.
    to get output do 
    Usage:
    -----
    s1=bmcmc._MH(model,['mu','sigma'])
    for i in range(10000):
        s1.next()        
    """
    def __init__(self,model,varnames,atime=10,ptime=1000):
        """
        Parameters        
        ----------
        model: a subclass of Model()
        varnames: a list of name of variables over which to do mcmc 
                  For each variable there must be a corresponding entry 
                  in Model().descr
        atime: iterations after which to adapt the proposal distribution        
        ptime: iterations after which to print summary on screen       
        """
        self.model=model
        self.varnames=varnames
        self.cov=np.diag(self.model.get_guess_variance(varnames))
        self.chain=[self.model.get_guess_values(varnames)]
        self.mu=self.chain[0]
        self.alpha=[0.0]
        self.model.lnprob_cur=self.model.lnprob(self.varnames,self.chain[0])
        self.lnprob=[np.sum(self.model.lnprob_cur)]
        
        self.atime=atime
        self.ptime=ptime
        self.lam=0.0
        self.i=0
#        self.j=1
        al=[0.0,0.44,0.352,0.316,0.279,0.275,0.266]
        self.alpha0=al[np.min([len(self.chain[0]),6])]
        self.delta=0.65



    def next(self):
        """
        advance mcmc to next step
        """
        y=np.random.multivariate_normal(np.array(self.chain[-1]),np.exp(self.lam)*self.cov,1)[0]

        if self.model.stoch:
            for i,name in enumerate(self.varnames):
                self.model.args[name]=self.chain[-1][i]
            self.model.refresh_stoch(self.model.args)
            lnprob_old=self.model.lnprob(self.varnames,self.chain[-1])
        else:
            lnprob_old=self.model.lnprob_cur

        lnprob_new=self.model.lnprob(self.varnames,y)

        self.alpha_cur=np.min([0,np.sum(lnprob_new)-np.sum(lnprob_old)])
        if np.log(np.random.ranf()) < self.alpha_cur:
            self.alpha.append(1.0)
            self.chain.append(y)
            self.lnprob.append(np.sum(lnprob_new))
            self.model.lnprob_cur=lnprob_new
        else:
            self.alpha.append(0.0)
            self.chain.append(self.chain[-1])
            self.lnprob.append(np.sum(lnprob_old))
        for i,name in enumerate(self.varnames):
            self.model.args[name]=self.chain[-1][i]

        self.i=self.i+1
        if self.atime>0:
            self._adapt()
        if (self.i)%self.ptime == 0:
            self._stats()

    def _adapt(self):
        #(i+1) needed to avoid getting stuck initially with cov=0 when alpha=0
        gamma=(1.0/(self.i+1))**self.delta
        self.lam=self.lam+(np.exp(self.alpha_cur)-self.alpha0)*gamma
        cov1=np.outer(self.chain[-1]-self.mu,self.chain[-1]-self.mu)
        self.cov=self.cov+(cov1-self.cov)*gamma
        self.mu=self.mu+(self.chain[-1]-self.mu)*gamma



    def _stats(self):
        """
        Print the current statistics. 
        First line: iteration, desired acceptance ratio, 
                    acceptance ratio for full sample, 
                    acceptance for last half, lambda, covariance  
        Second line: name,mean,stddev               
        """
        print '%-4s %10i %6.3f %6.3f %6.3f %6.3f'%('MH ',self.i,self.alpha0,np.mean(np.array(self.alpha)),np.mean(np.array(self.alpha)[-len(self.alpha)/2:]),self.lam),np.diag(np.exp(self.lam)*self.cov)
        x=np.array(self.chain)
        ntot=x.shape[0]
        for i in range(x.shape[1]):
            print '%16s %12g %12g'%(self.varnames[i],np.mean(x[:,i]),np.std(x[:,i]))

    def get_chain(self,burn=0,thin=1):
        """
        Get the mcmc output
        Parameters
        -------
        burn: the number of initial iterations to ignore
        thin: the no of itertionas to thin by
        Returns
        -------
        data: the mcmc chain for each variable as a dict
        """
        chain=np.array(self.chain)
        if burn>0:
            chain=chain[burn:]
        if thin>1:
            ind=np.arange(np.int(chain.shape[0]/thin))*thin
            chain=chain[ind]
        data={}
        for i,name in enumerate(self.varnames):
            data[name]=chain[:,i]
        return data

    def get_musigma(self):
        chain=np.array(self.chain[burn:])
        temp={}
        for i,name in enumerate(self.varnames):
            temp[name]=np.percentile(chain[:,i],[16.0,50.0,84.0])

    def info(self,burn=1000,plot=False):
        """
        Print the summary statistics and optionally plot the results
        """
        rows=len(self.varnames)
        cols=2
        chain=np.array(self.chain[burn:])
        nsize=chain.shape[0]
#        print rows,cols
        print '%4s %16s %12s %12s [%12s, %12s, %12s]'%('no','name','mean','stddev','16%','50%','84%')
        for i,name in enumerate(self.varnames):
            temp=np.percentile(chain[:,i],[16.0,84.0,50.0])
            print '%4i %16s %12g %12g [%12g, %12g, %12g]'%(i,name,np.mean(chain[:,i]),(temp[1]-temp[0])/2.0,temp[0],temp[2],temp[1])
            if plot:
                ax=plt.subplot(rows,cols,2*i+1) 
#                plt.text(0.05,0.9,r'$\tau$='+'%5.1f'%(acor.acor(chain[:,i])[0]),transform=ax.transAxes)
                plt.plot(chain[:,i])
                plt.ylabel(self.model.descr[name][3])
                plt.xlabel('Iteration')
                ax=plt.subplot(rows,cols,2*i+2) 
                plt.hist(chain[:,i],bins=100,histtype='step')
                plt.text(0.05,0.9,sround(np.mean(chain[:,i]),temp[0],temp[1]),transform=ax.transAxes)
                plt.xlabel(self.model.descr[name][3])
                # plt.text(0.05,0.9,'%6g %3g (%4g-%4g)'%(np.mean(chain[:,i]),(temp[1]-temp[0])/2.0,temp[0],temp[1]),transform=ax.transAxes)

    def write(self,filename,burn=0,thin=1,asdict=False):
        """
        Save information about the mcmc chain in an ebf file
        Parameters
        -------
        burn: the number of initial iterations to ignore
        thin: the no of itertionas to thin by
        asdict: save as a dict with keys as varnames
        """
        # if asdict:
        #     data=self.get_dict(burn,thin)
        #     ebf.write(filename,'/mh/chain/',data,'w')
        #     ebf.write(filename,'/h0/varnames',self.varnames,'a')
        #     ebf.write(filename,'/h0/alpha',np.array(self.alpha[burn:]),'a')
        # else:
        chain=np.array(self.chain)
        alpha=np.array(self.alpha)
        if burn>0:
            chain=chain[burn:]
            alpha=alpha[burn:]
        if thin>1:
            ind=np.arange(np.int(chain.shape[0]/thin))*thin
            chain=chain[ind]
            alpha=alpha[ind]
        ebf.write(filename,'/varnames',self.varnames,'w')
        ebf.write(filename,'/chain',chain,'a')
        ebf.write(filename,'/alpha',alpha,'a')

            

    # def read(filename,burn,thin):
    #     chain=ebf.read(filename,'/chain')
    #     chain=chain[burn:]
    #     ind=np.arange(np.int(chain.shape[0]/thin))*thin
    #     chain=chain[ind]
    #     return chain


