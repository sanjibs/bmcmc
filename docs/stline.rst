In bmcmc, a statistical model is specified by defining a
class derived from the parent class ``bmcmc.Model`` . The
user has to write 3 methods for this class.

* ``set_descr(self)``: This specifies the parameters of the
  model via a dictionary ``self.descr``.

* ``set_args(self)``: This specifies the data to be used by
  the model and also initializes all the variables that will be used to
  compute the posterior, via dictionary ``self.args``. The 
  parameters specified in ``self.descr``

* ``lnfunc(self,args)``: This is for specifying the
  likelihood function. 




Straight Line with outliers
---------------------------
The generative model is 

.. math::
   p(y_i| m, c, x_i, \sigma_{y,i}) = \frac{1}{\sqrt{2 \pi}
   \sigma_{y,i}}\exp\left(-\frac{(y_i - mx_i )^2}{2
   \sigma_{y,i}^2}\right) 

The background model is 

.. math::
   p(y_i|\mu_b,\sigma_b,x_i,\sigma_{y,i})=\frac{1}{\sqrt{2\pi(\sigma_{y,i}^2+\sigma_b^2)}}\exp\left(-\frac{(y_i-\mu_b)^2}{2 (\sigma_{y,i}^2+\sigma_b^2)}\right)

The full model is 

.. math::
   p(Y|m,c,\mu_b,P_b,\sigma_b,X,\sigma_y)=\prod_{i=1}^N [p(y_i|m,c,x_i,\sigma_{y,i})P_b+p(y_i|\mu_b,\sigma_b,x_i,\sigma_{y,i})(1-P_b)]


A model is constructed as follows::

    class stlineb(bmcmc.Model):
        def set_descr(self):
	    self.descr['m']      =['p', 1.0, 0.2,'$m$',       -1e10,1e10]
	    self.descr['c']      =['p',10.0, 1.0,'$c$',       -1e10,1e10]
	    self.descr['mu_b']   =['p', 1.0, 1.0,'$\mu_b$',   -1e10,1e10]
	    self.descr['sigma_b']=['p', 1.0, 1.0,'$\sigma_b$',1e-10,1e10]
	    self.descr['p_b']    =['p',0.15,0.01,'$P_b$',     1e-10,0.999]
	    self.descr['x']      =['d', 0.0, 1.0,'$x$',       -500.0,500.0]
	    self.descr['sigma_y']=['d', 1.0, 1.0,'$\sigma_x$',1e-10,1e3]
	    self.descr['y']      =['d', 0.0, 1.0,'$y$',       -500.0,500.0]

	 def set_args(self):
	     np.random.seed(11)
	     self.args['x']=0.5+np.random.ranf(self.dsize)*9.5
	     self.args['sigma_y']=0.25+np.random.ranf(self.dsize)
	     self.args['y']=np.random.normal(self.args['x']*2+10,self.args['sigma_y'])
	     self.ind=np.array([0,2,4,6,8,10,12,14,16,18])
	     self.args['y'][self.ind]=np.random.normal(30,5,self.ind.size)
	     self.args['y'][self.ind]=self.args['y'][self.ind]+np.random.normal(0.0,self.args['sigma_y'][self.ind])

	def lnfunc(self,args):
            temp1=(args['y']-(self.args['m']*self.args['x']+self.args['c']))/args['sigma_y']
	    sigma_b=np.sqrt(np.square(args['sigma_y'])+np.square(args['sigma_b']))
	    temp2=(args['y']-self.args['mu_b'])/sigma_b
	    temp1=temp1.clip(max=30.0)
	    temp2=temp2.clip(max=30.0)
	    temp11=(1-args['p_b'])*np.exp(-0.5*temp1*temp1)/(np.sqrt(2*np.pi)*args['sigma_y'])
	    temp22=args['p_b']*np.exp(-0.5*temp2*temp2)/(np.sqrt(2*np.pi)*sigma_b)
	    return np.log(temp11+temp22)
    
        def plot(self):
            plt.clf()
	    x = np.linspace(0,10)
	    plt.errorbar(self.args['x'], self.args['y'], yerr=self.args['sigma_y'], fmt=".k")
	    plt.errorbar(self.args['x'][self.ind], self.args['y'][self.ind], yerr=self.args['sigma_y'][self.ind], fmt=".r")
	    plt.plot(x,vals[0]*x+vals[1], color="g", lw=2, alpha=0.5)
	    plt.xlabel(r'$x$')
	    plt.ylabel(r'$y$')



The method set_descr is used to define the parameters of the model.


Hierarchical model using MWG
----------------------------


Exoplanets and binary system  
----------------------------
