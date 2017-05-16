import scipy.stats
import numpy as np
import matplotlib.pyplot  as plt
import bmcmc

# list the examples you want to run
ex_names=['gauss1','gauss2','stline','gmean']
ex_names=['gauss1','gauss2']

#------------------------------------------------------------
# A simple example: fitting a Gaussian function
class gauss1(bmcmc.Model):
    def set_descr(self):
        # setup descriptor
        # self.descr[varname]=[type, value, sigma,     latexname, min_val, max_val]
        self.descr['mu']     =['l0',   0.0,   1.0,      r'$\mu$',    -500,   500.0]
        self.descr['sigma']  =['l0',   1.0,   1.0,   r'$\sigma$',  1e-10,      1e3]

    def set_args(self):
        # setup data points 
        # Use self.args and self.eargs to define your variables. 
        # During sampling, the code will pass self.args as an argument to lnfunc.  
        np.random.seed(11)
        self.args['x']=np.random.normal(loc=0.0,scale=1.0,size=self.eargs['dsize'])

    def lnfunc(self,args):
        # define the function which needs to be sampled (log posterior)
        # Do not use self.args in this function. You can use self.eargs. 
        return scipy.stats.norm.logpdf(args['x'],loc=args['mu'],scale=args['sigma'])

if 'gauss1' in ex_names:
    # create an object
    m1=gauss1(eargs={'dsize':100})

    # run the sampler for two free parameters, for 50000
    # iterations and printing summary every 10000 iterations 
    m1.sample(['mu','sigma'],50000,ptime=10000)

    # print final results, discarding first 5000 iterations
    m1.info(burn=5000)

    # plot final results , discarding first 5000 iterations
    plt.figure()
    m1.plot(burn=5000)

    # print chain values for a few given parameters
    print m1.chain['mu']
    print m1.chain['sigma']

    # To print final results as a latex table, discarding the first 5000 iterations.
    m1.info(burn=5000,latex=True)

    # plot final results for specific parameters
    #mymodel.plot(keys=['mu','sigma'],burn=1000)
    plt.savefig('../docs/images/gauss1.png')
#------------------------------------------------------------
# Hierarchical Bayesian model.
# Fitting a Gaussian function taking into account obsevational 
# uncertainties in data.

class gauss2(bmcmc.Model):
   def set_descr(self):
       # setup descriptor
       self.descr['mu']     =['l0',0.0,1.0,r'$\mu$'   ,-500,500.0]
       self.descr['sigma']  =['l0',1.0,1.0,r'$\sigma$',1e-10,1e3]
       self.descr['xt']     =['l1',0.0,1.0,r'$x_t$'   ,-500.0,500.0]

   def set_args(self):
       # setup data points 
       np.random.seed(11)
       # generate true coordinates of data points
       self.args['x']=np.random.normal(loc=self.eargs['mu'],scale=self.eargs['sigma'],size=self.eargs['dsize'])
       # add observational uncertainty to each data point
       self.args['sigma_x']=np.zeros(self.args['x'].size,dtype=np.float64)+0.5
       self.args['x']=np.random.normal(loc=self.args['x'],scale=self.args['sigma_x'],size=self.eargs['dsize'])

   def lnfunc(self,args):
       # log posterior
       temp1=scipy.stats.norm.logpdf(args['xt'],loc=args['mu'],scale=args['sigma'])
       temp2=scipy.stats.norm.logpdf(args['x'],loc=args['xt'],scale=args['sigma_x'])
       return temp1+temp2

if 'gauss2' in ex_names:
    m1=gauss2(eargs={'dsize':100,'mu':0.0,'sigma':1.0})
    m1.sample(['mu','sigma','xt'],50000,ptime=10000)
    m1.info()
    plt.figure()
    m1.plot(burn=5000)
    #For level-1 parameters only mean and stddev are stored  and made available.
    # print mean value of xt for each data point
    print m1.mu['xt']
    # print stddev value of xt for each data point
    print m1.sigma['xt']
    #plt.savefig('../docs/images/gauss2.png')



#------------------------------------------------------------
# straight line with outliers
class stlineb(bmcmc.Model):
    def set_descr(self):
        # setup descriptor
        self.descr['m']      =['l0', 1.0, 0.2,'$m$',       -1e10,1e10]
        self.descr['c']      =['l0',10.0, 1.0,'$c$',       -1e10,1e10]
        self.descr['mu_b']   =['l0', 1.0, 1.0,'$\mu_b$',   -1e10,1e10]
        self.descr['sigma_b']=['l0', 1.0, 1.0,'$\sigma_b$',1e-10,1e10]
        self.descr['p_b']    =['l0',0.1,0.01,'$P_b$',       1e-10,0.999]

    def set_args(self):
        # setup data points 
        np.random.seed(11)
        self.args['x']=0.5+np.random.ranf(self.eargs['dsize'])*9.5
        self.args['sigma_y']=0.25+np.random.ranf(self.eargs['dsize'])
        self.args['y']=np.random.normal(self.args['x']*2+10,self.args['sigma_y'])
        # add outliers
        self.ind=np.array([0,2,4,6,8,10,12,14,16,18])
        self.args['y'][self.ind]=np.random.normal(30,5,self.ind.size)
        self.args['y'][self.ind]=self.args['y'][self.ind]+np.random.normal(0.0,self.args['sigma_y'][self.ind])

    def lnfunc(self,args):
        if self.eargs['outliers'] == False:
            temp1=(args['y']-(self.args['m']*self.args['x']+self.args['c']))/args['sigma_y']
            return -0.5*(temp1*temp1)-np.log(np.sqrt(2*np.pi)*args['sigma_y'])
        else:
            temp11=scipy.stats.norm.pdf(args['y'],loc=(self.args['m']*self.args['x']+self.args['c']),scale=args['sigma_y'])
            sigma_b=np.sqrt(np.square(args['sigma_y'])+np.square(args['sigma_b']))
            temp22=scipy.stats.norm.pdf(args['y'],loc=self.args['mu_b'],scale=sigma_b)
            return np.log((1-args['p_b'])*temp11+args['p_b']*temp22)

    def myplot(self,chain): 
       # optional for plotting
        plt.clf()
        x = np.linspace(0,10)
        burn=self.chain['m'].size/2
        vals=self.best_fit(burn=burn)
        plt.errorbar(self.args['x'], self.args['y'], yerr=self.args['sigma_y'], fmt=".k")
        plt.errorbar(self.args['x'][self.ind], self.args['y'][self.ind], yerr=self.args['sigma_y'][self.ind], fmt=".r")
        plt.plot(x,vals[0]*x+vals[1], color="g", lw=2, alpha=0.5)
        for i,key in enumerate(self.names0):
            print key
            plt.text(0.5,0.3-i*0.06,self.descr[key][3]+'='+bmcmc.stat_text(self.chain[key][burn:]),transform=plt.gca().transAxes)

        vals1=[]
        burn1=chain['m'].size/2
        for i,key in enumerate(['m','c']):
            print key
            plt.text(0.05,0.5-i*0.05,self.descr[key][3]+'='+bmcmc.stat_text(chain[key][burn1:]),transform=plt.gca().transAxes)
            vals1.append(np.mean(chain[key][burn1:]))
        plt.plot(x,vals1[0]*x+vals1[1], 'g--', lw=2, alpha=0.5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.axis([0,10,5,40,])



# Expected values are 
#('m','c','p_b','mu_b','sigma_b')=(2.0, 10.0, 0.2, 30, 5.0')
if 'stline' in ex_names:
    model1=stlineb(eargs={'dsize':50})
    model1.eargs['outliers']=False
    model1.sample(['m','c'],10000)
    chain1=model1.chain
    
    model1.eargs['outliers']=True
    model1.sample(['m','c','p_b','mu_b','sigma_b'],20000)
    plt.figure()
    model1.myplot(chain1)
    plt.savefig('../docs/images/stline.png')

#------------------------------------------------------------
class gauss3(bmcmc.Model):
   def set_descr(self):
       # setup descriptor
       self.descr['mu']     =['l0',0.0,1.0,r'$\mu$ ',-500,500.0]
       self.descr['sigma']  =['l0',1.0,1.0,r'$\sigma$ ',1e-10,1e3]

   def set_args(self):
       # setup data points 
       np.random.seed(11)
       # generate true coordinates of data points
       self.args['x']=np.random.normal(loc=self.eargs['mu'],scale=self.eargs['sigma'],size=self.eargs['dsize'])
       # add observational uncertainty to each data point
       self.args['sigma_x']=np.zeros(self.args['x'].size,dtype=np.float64)+0.5
       self.args['x']=np.random.normal(loc=self.args['x'],scale=self.args['sigma_x'],size=self.eargs['dsize'])

   def lnfunc(self,args):
       # log posterior, xt has been integrated out
       sigma=np.sqrt(args['sigma']*args['sigma']+args['sigma_x']*args['sigma_x'])
       temp=scipy.stats.norm.logpdf(args['x'],loc=args['mu'],scale=sigma)
       return temp

if 'da' in ex_names:
    # Data augmentation, making use of Hierarchical modelling
    # marginalization using sampling.
    m2=gauss2(eargs={'dsize':1000,'mu':0.0,'sigma':1.0})
    m2.sample(['mu','sigma','xt'],100000,ptime=10000)
    # Marginalization using direct integration
    m3=gauss3(eargs={'dsize':1000,'mu':0.0,'sigma':1.0})
    m3.sample(['mu','sigma'],100000,ptime=10000)

    plt.subplot(2,2,1)
    plt.hist(m2.chain['mu'],range=[-0.2,0.2],bins=100,normed=True,histtype='step',lw=2.0)
    plt.hist(m3.chain['mu'],range=[-0.2,0.2],bins=100,normed=True,histtype='step',lw=2.0)
    plt.ylabel('p')
    plt.xlabel(r'$\mu$')
    plt.xlim([-0.2,0.2])
    plt.xticks([-0.2,-0.1,0.0,0.1,0.2])

    plt.subplot(2,2,2)
    plt.hist(m2.chain['sigma'],range=[0.85,1.15],bins=100,normed=True,histtype='step',lw=2.0)
    plt.hist(m3.chain['sigma'],range=[0.85,1.15],bins=100,normed=True,histtype='step',lw=2.0)
    plt.ylabel('p')
    plt.xlabel(r'$\sigma$')
    plt.xlim([0.85,1.15])
    plt.xticks([0.9,1.0,1.1])

    plt.subplot(2,2,3)
    nsize=50
    plt.plot(np.arange(nsize),bmcmc.autocorr(m2.chain['mu'])[0:nsize],label='DA',lw=2.0)
    plt.plot(np.arange(nsize),bmcmc.autocorr(m3.chain['mu'])[0:nsize],label='Integration',lw=2.0)
    plt.ylabel(r'$\rho_{\mu \mu}(t)$')
    plt.xlabel(r'lag $t$')
    plt.legend()
    plt.axis([0,50,0,1])

    plt.subplot(2,2,4)
    plt.plot(np.arange(nsize),bmcmc.autocorr(m2.chain['sigma'])[0:nsize],lw=2.0)
    plt.plot(np.arange(nsize),bmcmc.autocorr(m3.chain['sigma'])[0:nsize],lw=2.0)
    plt.ylabel(r'$\rho_{\sigma \sigma}(t)$')
    plt.xlabel(r'lag $t$')
    plt.axis([0,50,0,1])
    plt.tight_layout()
    #plt.savefig('../docs/images/da_demo1.png')


#------------------------------------------------------------------------
class gmean(bmcmc.Model):
    def set_descr(self):
        # Setup descriptor.
        self.descr['alpha']   =['l1',0.0,1.0,r'$\alpha$',-1e10,1e10]
        self.descr['mu']      =['l0',1.0,1.0,r'$\mu$'   ,-1e10,1e10]
        self.descr['omega']   =['l0',1.0,1.0,r'$\omega$',1e-10,1e10]

    def set_args(self):
        # Create data points.
        np.random.seed(11)
        self.eargs['mu']=0.0
        self.eargs['omega']=1.0
        self.eargs['sigma']=1.0

        self.data={}
        self.data['y']=[]
        self.data['gsize']=np.array([2,4,6,8,10]*(self.eargs['dsize']/5))
        self.data['gmean']=np.random.normal(self.eargs['mu'],self.eargs['omega'],size=self.data['gsize'].size)
        for i in range(self.data['gsize'].size):
            self.data['y'].append(np.random.normal(self.data['gmean'][i],self.eargs['sigma'],size=self.data['gsize'][i]))

    def lnfunc(self,args):
       # log posterior
        temp1=[]
        for i,y in enumerate(self.data['y']):
            temp1.append(np.sum(self.lngauss(y-args['alpha'][i],self.eargs['sigma'])))
        temp1=np.array(temp1)
        temp2=scipy.stats.norm.logpdf(args['alpha'],loc=args['mu'],scale=args['omega'])
        return temp1+temp2

    def myplot(self):
        # Plot the results
        plt.clf()
        burn=1000
        x=np.arange(self.eargs['dsize'])+1
        stats=[[],[]]
        for i,y in enumerate(self.data['y']):
            stats[0].append(np.mean(y))
            stats[1].append(self.eargs['sigma']/np.sqrt(y.size))
        plt.errorbar(x,stats[0],yerr=stats[1],fmt='.b',lw=3,ms=12,alpha=0.8) 
        plt.errorbar(x,self.mu['alpha'],yerr=self.sigma['alpha'],fmt='.g',lw=3,ms=12,alpha=0.8)

        temp1=np.mean(self.chain['mu'][burn:])
        plt.plot([0,self.eargs['dsize']+1],[temp1,temp1],'k--')
        plt.xlim([0,self.eargs['dsize']+1])
        plt.ylabel(r'$\alpha_j$')
        plt.xlabel(r'Group $j$')

if 'gmean' in ex_names:
    m1=gmean(eargs={'dsize':40})
    m1.sample(['mu','omega','alpha'],10000)
    m1.myplot()
    #plt.savefig('../docs/images/gmean.png')
