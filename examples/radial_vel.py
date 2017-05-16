import numpy as np
import scipy.stats
import scipy.interpolate
import matplotlib.pyplot as plt
import bmcmc

# functions for computing the radial velocity curve
def true_anomaly(t,tp,e,tau):
    temp1=np.min((t-tau)/tp)-1
    temp2=np.max((t-tau)/tp)+1
    u1=np.linspace(2*np.pi*temp1,2*np.pi*temp2,1000)
    ma=u1-e*np.sin(u1)
    myfunc=scipy.interpolate.interp1d(ma,u1)
    u=myfunc((2*np.pi)*(t-tau)/tp)
    return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(0.5*u))

def kappa(t,e,m,mc,i):
    au_yr=4.74057 # km/s
    G=4*np.pi*np.pi*np.power(au_yr,3.0) # [km/s]^{3}*[yr]^{-1}*[M_sol]
    nr=np.power(2*np.pi*G,1/3.0)*mc*np.sin(np.radians(i))
    dr=np.power(t/365.0,1/3.0)*np.power(m+mc,1/3.0)*np.sqrt(1-e*e)
    return nr/dr

def vr(t,kappa,tp,e,tau,omega,v0):
    omega=np.radians(omega)
    f=true_anomaly(t,tp,e,tau)
    return kappa*(np.cos(f+omega)+e*np.cos(omega))+v0

# model for describing a binary system
class binary_model(bmcmc.Model):
    def set_descr(self):
        # setup descriptor
        self.descr['kappa']     =['p',0.1,0.1,r'$\kappa$ ',1e-10,10.0]
        self.descr['tp']  =['p',365.0,10.0,r'$T$ ',1e-10,1e6]
        self.descr['e']      =['p',0.5,0.1,r'$e$ ',0,1.0]
        self.descr['tau']     =['p',50.0,10.0,r'$\tau$ ',-360.0,360.0]
        self.descr['omega']=['p',180.0,10.0,r'$\omega$ ',-360,360.0]
        self.descr['v0']=['p',0.0,1.0,r'$v_0$ ',-1e3,1e3]
        self.descr['s']=['p',0.02,1.0,r'$\sigma$ ',1e-10,1e3]

    def set_args(self):
        # setup data points 
        np.random.seed(17)
        kappa=0.15
        tp=350.0
        e=0.3
        tau=87.5
        omega=-90.0
        v0=0.0
        print tau,omega
        
        self.args['t']=np.linspace(0,self.args['tp']*1.5,self.eargs['dsize'])
        vr1=vr(self.args['t'],kappa,tp,e,tau,omega,v0)
        self.args['vr']=vr1+np.random.normal(0.0,self.args['s'],size=self.eargs['dsize'])
    

    def lnfunc(self,args):
        vr1=vr(args['t'],args['kappa'],args['tp'],args['e'],args['tau'],args['omega'],args['v0'])
        temp=-np.square(args['vr']-vr1)/(2*args['s']*args['s'])-np.log(np.sqrt(2*np.pi)*args['s'])
        return temp



# Create an object of the model.
m1=binary_model(eargs={'dsize':50})

# Run the sampler.
m1.sample(['kappa','tp','e','tau','omega','v0'],50000,ptime=10000)

# Plot the results.
plt.figure()
m1.plot(keys=['kappa','tp','e'],burn=10000)
plt.tight_layout()
# plt.savefig('../docs/images/rv_mcmc_params.png')

# Plot the best fit model. 
res=m1.best_fit()
args=m1.args
t=np.linspace(0,np.max(m1.args['t']),1000)
plt.figure()
plt.errorbar(args['t'],args['vr'],yerr=args['s'],fmt='o',label='data')
plt.plot(t,vr(t,res[0],res[1],res[2],res[3],res[4],res[5]),label='best fit',lw=2.0)
plt.plot(t,vr(t,res[0],res[1],0.0,res[3],res[4],res[5]),'r--',label='e=0.0',lw=1.0)
plt.title(r'$\kappa=0.15$ km/s, $T=350$ days, $e=0.3$, $\omega=-90.0^{\circ}$, $\tau=87.5$ days')
plt.xlabel('Time (days)')
plt.ylabel('radial velocity (km/s)')
plt.legend(loc='lower left',frameon=False)
plt.axis([0,600,-0.35,0.35])
plt.tight_layout()
# plt.savefig('../docs/images/rv_mcmc.png')

