# translated from /u/samn/npredict/geom_mainen.hoc // $Id: geom_mainen.hoc,v 1.3 2007/04/16 14:42:33 samn Exp $
from neuron import h
from math import pi

class ITS4_cell:
  def __init__ (self,ID=0,ty=0,col=0,rho=165.0,kappa=0.01,soma_pas=True): #kappa =10 --> 0 is fully connected 
    self.ID=ID
    self.ty=ty
    self.col=col
    self.soma_pas = soma_pas
    self.soma = soma = h.Section(name='soma',cell=self)
    self.dend = dend = h.Section(name='dend',cell=self)
    self.dend.connect(self.soma,0.5,0) #   connect dend(0), soma(0.5)
    self.rho = rho # dendritic to axo-somatic area ratio 
    self.kappa = kappa # coupling resistance (Mohm) 
    for sec in [self.soma,self.dend]:
      sec.insert('k_ion')
      sec.insert('na_ion')
      sec.insert('ca_ion') #PROBLEMS??
      sec.ek = -90 # K+ current reversal potential (mV)
      sec.ena = 60 # Na+ current reversal potential (mV)
      sec.eca = 140 # Ca2+ current reversal potential (mV)    ## PROBLEMS FROM LINE 19  
      h.ion_style("ca_ion",0,1,0,0,0) # using an ohmic current rather than GHK equation
      sec.Ra=100 #dend.Ra = 30 #kilo-ohms -- cm2
    self.initsoma()
    self.initdend()

  def initsoma (self):
    soma = self.soma
    soma.nseg = 1
    soma.diam = 10.0/pi
    soma.L = 10
    soma.cm = 0.75
    soma.insert('naz') # naz.mod
    soma.insert('kv') # kv.mod
    soma.gmax_naz = 30e3
    soma.gbar_kv = 1.5e3 ## gbar_kv not gmax_kv in OLD mod file (check nrniv/mod?)
    if self.soma_pas:
      soma.insert('pas')
      soma.e_pas=-70
      soma.g_pas=1/3e4

  def initdend (self):
    dend = self.dend
    dend.nseg = 1
    dend.diam = 10.0/pi
    self.config()
    dend.cm = 0.75 # microfarads / cm2
    dend.insert('naz') # naz.mod
    dend.insert('km') # km.mod
    dend.insert('kca') # kca.mod
    dend.insert('Nca') # Nca.mod
    dend.insert('cadad') # cadad.mod
    dend.insert('pas')
    dend.eca=140
    h.ion_style("ca_ion",0,1,0,0,0) # already called before
    dend.e_pas = -70 # only dendrite has leak conductance - why?
    dend.g_pas = 1/3e4 # only dendrite has leak conductance
    dend.gmax_naz=15
    dend.gmax_Nca = 0.3 # high voltage-activated Ca^2+ 
    dend.gbar_km = 0.1 ## gbar_km not gmax_km (check new nrniv/mod??) # slow voltage-dependent non-inactivating K+ 
    dend.gbar_kca = 3  ## gbar_kca not gmax (check new nrniv/mod??) # slow Ca^2+-activated K+

  def config (self):
    self.dend.L  = self.rho*self.soma.L  #535 #545 - soma.L (10)#self.rho*self.soma.L # dend area is axon area multiplied by rho
    self.dend.Ra = self.dend.Ra*self.kappa/self.dend(0.5).ri() # axial resistivity is adjusted to achieve

  # resets cell to default values
  def todefault(self):
    self.rho = 165
    self.kappa = 10
    self.config()
    self.soma.gmax_naz = 30e3
    self.soma.gbar_kv = 1.5e3 ##see note above 
    self.dend.g_pas = 1/3e4
    self.dend.gmax_naz = 15
    self.dend.gmax_Nca = 0.3
    self.dend.gbar_km = 0.1 ## see note above
    self.dend.gbar_kca = 3 ## see note above 
    self.soma.ek = dend.ek = -90
    self.soma.ena = dend.ena = 60
    self.soma.eca = dend.eca = 140
