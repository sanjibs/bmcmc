In bmcmc, we fit a statistical model specified by a set of parameters :math:`\theta` to some data :math:`D`. 
Using MCMCM we sample the posterior :math:`p(\theta|D)`. A statistical model is specified by defining a
class derived from the parent class ``bmcmc.Model`` . The user has to write 3 methods for this class.

* ``set_descr(self)``: This sets the dictionary ``self.descr`` that describes the parameters :math:`\theta` of the
  model.
  Example::
  
  self.descr[name] =[level, value, sigma, latex_name, value_min, value_max]  
  self.descr['m']  =['l0' , 1.0  , 0.2  , '$m$'     , -1e10    , 1e10     ]
  
  For a hierarchical model parameters can exist at various levels. We have two options to choose from 'l0' or 'l1'.  'l0'denotes the top level, for example the hyperparameters, and 'l1' denotes the level below it.  
  
* ``set_args(self)``: This sets the dictionary ``self.args`` that specifies the data :math:`D` to be used by
  the model. Any parameters :math:`\theta` or variables that will be used to
  compute the posterior, should also be initialized here. Any remaining   
  uninitialized parameters from ``self.descr`` are automatically initialized.

* ``lnfunc(self,args)``: This is for specifying the
  posterior probability :math:`p(\theta|D)` that we want to sample using MCMC. This could either be a scalar or an array.

Here we discuss the simple case of fitting a straight line to some data points :math:`D=\{(x_1,y_1),(x_2,y_2)...(x_N,y_N)\}`


Straight Line with outliers
---------------------------
The straight line model to decribe the points is  

.. math::
   p(y_i| m, c, x_i, \sigma_{y,i}) = \frac{1}{\sqrt{2 \pi}
   \sigma_{y,i}}\exp\left(-\frac{(y_i - mx_i )^2}{2
   \sigma_{y,i}^2}\right) 

The background model is 

.. math::
   p(y_i|\mu_b,\sigma_b,x_i,\sigma_{y,i})=\frac{1}{\sqrt{2\pi(\sigma_{y,i}^2+\sigma_b^2)}}\exp\left(-\frac{(y_i-\mu_b)^2}{2 (\sigma_{y,i}^2+\sigma_b^2)}\right)

The full model to describe the data is 

.. math::
   p(Y|m,c,\mu_b,P_b,\sigma_b,X,\sigma_y)=\prod_{i=1}^N [p(y_i|m,c,x_i,\sigma_{y,i})P_b+p(y_i|\mu_b,\sigma_b,x_i,\sigma_{y,i})(1-P_b)]


A model is constructed as follows::

    class stlineb(bmcmc.Model):
        def set_descr(self):
	    self.descr['m']      =['l0', 1.0, 0.2,'$m$',       -1e10,1e10]
	    self.descr['c']      =['l0',10.0, 1.0,'$c$',       -1e10,1e10]
	    self.descr['mu_b']   =['l0', 1.0, 1.0,'$\mu_b$',   -1e10,1e10]
	    self.descr['sigma_b']=['l0', 1.0, 1.0,'$\sigma_b$',1e-10,1e10]
	    self.descr['p_b']    =['l0',0.15,0.01,'$P_b$',     1e-10,0.999]
	    self.descr['x']      =['l1', 0.0, 1.0,'$x$',       -500.0,500.0]
	    self.descr['sigma_y']=['l1', 1.0, 1.0,'$\sigma_x$',1e-10,1e3]
	    self.descr['y']      =['l1', 0.0, 1.0,'$y$',       -500.0,500.0]

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
