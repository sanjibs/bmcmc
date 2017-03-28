import numpy as np
import matplotlib.pyplot  as plt
import bmcmc

# straight line 
class stline(bmcmc.Model):
    def set_descr(self):
        # setup descriptor
        self.descr['m']      =['p',0.0,1.0,'$m$',-1e10,1e10]
        self.descr['c']      =['p',1.0,1.0,'$c$',-1e10,1e10]
        self.descr['x']      =['d',0.0,1.0,'$x$',-500.0,500.0]
        self.descr['sigma_y']=['d',1.0,1.0,'$\sigma_x$',1e-10,1e3]
        self.descr['y']      =['d',0.0,1.0,'$y$',-500.0,500.0]
        

    def set_args(self):
        # setup data points 
        np.random.seed(11)
        self.args['x']=0.5+np.random.ranf(self.eargs['dsize'])*9.5
        self.args['sigma_y']=0.25+np.random.ranf(self.eargs['dsize'])
        self.args['y']=np.random.normal(self.args['x']*2+10,self.args['sigma_y'])
        # self.ind=np.array([0,2,4,6,8,10,12,14,16,18])
        # self.args['y'][self.ind]=np.random.normal(30,5,self.ind.size)
        # self.args['y'][self.ind]=self.args['y'][self.ind]+np.random.normal(0.0,self.args['sigma_y'][self.ind])

    def lnfunc(self,args):
        # setup the lnprob function 
        temp=(args['y']-(self.args['m']*self.args['x']+self.args['c']))/args['sigma_y']
        return -0.5*temp*temp-np.log(np.sqrt(2*np.pi)*args['sigma_y'])

    def plot(self,vals=None):
        # optional for plotting
        vals=[self.mu['m'],self.mu['c']]
        plt.clf()
        x = np.linspace(0,10)
        plt.errorbar(self.args['x'], self.args['y'], yerr=self.args['sigma_y'], fmt=".k")
        plt.errorbar(self.args['x'][self.ind], self.args['y'][self.ind], yerr=self.args['sigma_y'][self.ind], fmt=".r")
        plt.plot(x,vals[0]*x+vals[1], color="g", lw=2, alpha=0.5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')

# straight line with outliers
class stlineb(bmcmc.Model):
    def set_descr(self):
        # setup descriptor
        self.descr['m']      =['p', 1.0, 0.2,'$m$',       -1e10,1e10]
        self.descr['c']      =['p',10.0, 1.0,'$c$',       -1e10,1e10]
        self.descr['mu_b']   =['p', 1.0, 1.0,'$\mu_b$',   -1e10,1e10]
        self.descr['sigma_b']=['p', 1.0, 1.0,'$\sigma_b$',1e-10,1e10]
        self.descr['p_b']    =['p',0.15,0.01,'$P_b$',     1e-10,0.999]
        self.descr['x']      =['d', 0.0, 1.0,'$x$',       -500.0,500.0]
        self.descr['sigma_y']=['d', 1.0, 1.0,'$\sigma_x$',1e-10,1e3]
        self.descr['y']      =['d', 0.0, 1.0,'$y$',       -500.0,500.0]

    def set_args(self):
        # setup data points 
        np.random.seed(11)
        self.args['x']=0.5+np.random.ranf(self.eargs['dsize'])*9.5
        self.args['sigma_y']=0.25+np.random.ranf(self.eargs['dsize'])
        self.args['y']=np.random.normal(self.args['x']*2+10,self.args['sigma_y'])
        self.ind=np.array([0,2,4,6,8,10,12,14,16,18])
        self.args['y'][self.ind]=np.random.normal(30,5,self.ind.size)
        self.args['y'][self.ind]=self.args['y'][self.ind]+np.random.normal(0.0,self.args['sigma_y'][self.ind])

    def lnfunc(self,args):
        # setup the lnprob function 
        temp1=(args['y']-(self.args['m']*self.args['x']+self.args['c']))/args['sigma_y']
        sigma_b=np.sqrt(np.square(args['sigma_y'])+np.square(args['sigma_b']))
        temp2=(args['y']-self.args['mu_b'])/sigma_b
        temp1=temp1.clip(max=30.0)
        temp2=temp2.clip(max=30.0)
        temp11=(1-args['p_b'])*np.exp(-0.5*temp1*temp1)/(np.sqrt(2*np.pi)*args['sigma_y'])
        temp22=args['p_b']*np.exp(-0.5*temp2*temp2)/(np.sqrt(2*np.pi)*sigma_b)
        return np.log(temp11+temp22)

    def plot(self):
        # optional for plotting
        plt.clf()
        x = np.linspace(0,10)
        # for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
        #     plt.plot(xl, m*xl+b, color="k", alpha=0.1)
        plt.errorbar(self.args['x'], self.args['y'], yerr=self.args['sigma_y'], fmt=".k")
        plt.errorbar(self.args['x'][self.ind], self.args['y'][self.ind], yerr=self.args['sigma_y'][self.ind], fmt=".r")
        plt.plot(x,vals[0]*x+vals[1], color="g", lw=2, alpha=0.5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')


class gauss1(bmcmc.Model):
    def set_descr(self):
        # setup descriptor
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
        temp1=-np.square(args['xt']-args['mu'])/(2*args['sigma']*args['sigma'])-np.log(np.sqrt(2*np.pi)*args['sigma'])
        temp=temp1-np.square(args['x']-args['xt'])/(2*args['sigma_x']*args['sigma_x'])-np.log(np.sqrt(2*np.pi)*args['sigma_x'])
        return temp




#------------------------------------------------------------
model1=stline(eargs={'dsize':50})
model1.sample(['m','c'],10000,ptime=20000)
print 'Expected values are, 2.0, 10.0'
model1.info()

#------------------------------------------------------------
# straight line with outliers
model2=stlineb(eargs={'dsize':50})
model2.sample(['m','c','p_b','mu_b','sigma_b'],10000,ptime=20000)
print 'Expected values are, 2.0, 10.0, 0.2, 30, 5.0'
model2.info()

#------------------------------------------------------------
model=gauss1(eargs={'dsize':50})
model.sample(['xt','mu','sigma'],10000)
model.info()
