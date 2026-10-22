"""Bounded streaming STFT and true peak/RMS envelopes for the editor timeline."""
import numpy as np

class AudioSpectrum:
 def __init__(self,duration,bins=4096,rate=16000,bands=40):
  bins=min(bins,max(1,int(np.ceil(duration*rate/256))));self.bins=bins;self.rate=rate;self.total=max(1,round(duration*rate));self.offset=0
  self.peaks=np.zeros(bins);self.energy=np.zeros(bins);self.count=np.zeros(bins)
  self.power=np.zeros((bins,bands));self.windows=np.zeros(bins)
  self.tail=np.zeros(0,np.float32);self.tail_at=0;self.window=np.hanning(512)
  frequencies=np.fft.rfftfreq(512,1/rate);edges=np.geomspace(40,rate/2,bands+1)
  self.indices=[np.flatnonzero((frequencies>=a)&(frequencies<b)) for a,b in zip(edges,edges[1:])]
  self.indices=[ids if len(ids) else np.array([min(len(frequencies)-1,max(1,round((a+b)/2*512/rate)))]) for ids,a,b in zip(self.indices,edges,edges[1:])]
 def add(self,values):
  values=np.nan_to_num(values,nan=0,posinf=0,neginf=0)
  columns=np.minimum(self.bins-1,(np.arange(len(values))+self.offset)*self.bins//self.total)
  np.maximum.at(self.peaks,columns,np.abs(values));np.add.at(self.energy,columns,values.astype(float)**2);np.add.at(self.count,columns,1)
  samples=np.concatenate((self.tail,values));position=0
  while position+512<=len(samples):
   power=np.abs(np.fft.rfft(samples[position:position+512]*self.window))**2/(self.window.sum()**2)
   column=min(self.bins-1,(self.tail_at+position+256)*self.bins//self.total)
   self.power[column]+=[float(power[ids].mean()) for ids in self.indices];self.windows[column]+=1;position+=256
  self.tail=samples[position:].copy();self.tail_at+=position;self.offset+=len(values)
 def result(self):
  spectrum=self.power/np.maximum(1,self.windows[:,None]);db=10*np.log10(np.maximum(spectrum,1e-10))
  levels=np.clip((db+80)/70,0,1)*255
  return {'peaks':np.clip(self.peaks,0,1).round(4).tolist(),'rms':np.sqrt(self.energy/np.maximum(1,self.count)).round(4).tolist(),'spectrum':levels.astype(np.uint8).tolist(),'frequency_min':40,'frequency_max':self.rate//2,'fft_size':512,'sample_rate':self.rate}
