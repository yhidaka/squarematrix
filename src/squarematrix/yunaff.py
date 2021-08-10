import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import copy
from numpy import array
import pickle
#This yunaff 
#1. uses naff with hanning window in projection maximization of <nu|v>
#2. select tune peaks|nu> to be Gramm-Schidt orthigonalized using three points with middle point higher as criterion for peak selection
#3. interpolated_fft uses range only in rng to choose peak.
#4. naffref use reference frequencies to build Gramm-Schmidt basis and disregards other peaks not in near the fref.
#5. naff sorts out largest peaks in fft as basis to build Gramm-Schmidt basis
#6. yufminscaled uses fmin but with zero point shifted and variable scaled first before using fmin.
def sva(filename,x):
    ff=open(filename,'a')
    pickle.dump(x,ff)
    ff.close()


def sv(filename,x):
        ff=open(filename,'wb')
        pickle.dump(x,ff)
        ff.close()

def interpolated_fft(xi,rng=[0.,1],ith=1,hann=True,
               verbose=False,semilog=False,
               figsize=(10,3)):
    '''
    get the fractional tune by naff, with peaks selected as any point with two sides with smaller amplitude, then the peaks are sorted
    with the maximum chosen and fit with 3 point formula for interpolated tune
    rng: the fractual range of tune range in which tune is desired
    x: 1d array-like TbT data
    ith: which peak to be choosen, 1st one by default
    return tune fractional

    '''
    """
    The following is copied from naff.f90
    #Parabolic Interpolation (use with hanning window)
    #lk = fft_amp(max_ix)
    #lkm = fft_amp(max_ix-1)
    #lkp = fft_amp(max_ix+1)
    #A = (lkp-lkm) / 2.0 / (2.0*lk - lkp - lkm)
    #interpolated_fft = 1.0d0*max_ix/n_samples + A/n_samples
    """
    #xi=v20
    x = copy.deepcopy(xi)
    #x -= np.average(x)
    #rng=[0.056,0.06]
    T = len(x)
    nmf = T-1 # normalization factor
    #print ('T=',T,'type(T)',type(T))
    if hann:
        hw = np.arange(T)*1.0/(T-1)
        hw = (1-np.cos(2*np.pi*hw))
        x *= hw #x=>x*hw
    xpw = np.abs(np.fft.fft(x))
    rngi=list(map(int, T*np.array(rng)))
    dasp=[   [j, j*1.0/T,  asp  ] for j,asp in enumerate(xpw) if 0 < j < T-1 and rngi[0]<=j<=rngi[1] and xpw[j-1]<xpw[j]>xpw[j+1] ]
    if xpw[0]>xpw[1]: dasp.insert(0,[0,0.0,xpw[0]])
    if xpw[len(xpw)-1]>xpw[len(xpw)-2] : dasp.append([len(xpw),len(xpw)*1.0/T,xpw[len(xpw)-1]])
    dasp=np.array(dasp).transpose()#select all points with two points next to it having amplitudes smaller than the middle point
    idxdasp=np.argsort(dasp[2])
    idxpeaks=dasp[0][idxdasp]#the positions in the fft spectrum where middle point is larger than its two sides, sorted according to the differences from the sides
    tunesorted=idxpeaks/T #The tune of the sorted peaks
    #print "tunesorted[-5:]=\n",tunesorted[-5:]
    maxi =int(idxpeaks[-1])#the position of the point with maximum of smaller difference with its neighbor
    lk =xpw[maxi]
    lkm =xpw[maxi-1]
    lkp = xpw[maxi+1]
    A = (lkp-lkm) / 2.0 / (2.0*lk - lkp - lkm)
    peaktune = (maxi + A)/T
    return  peaktune,tunesorted[-20:]

def naffref(f,fref=[0.058,0.065,0.051],df=2e-3): #naffref use specified tunes fref to find peaks tune within df of fref, eg. 0.058-df<nu<0.058+df
        '''
    NAFF: J Laskar, Physica D 56 (1992) 253-269,LascarFMA.pdf
    f: tbt data
    ni: total number of Gramm-Schmidt orthogonalizations
    returns:
    nfnu: tune peaks
    amp: absolute amplitudes
    nran: to avoid local min, try nran times random initial phase
    '''
        fc = copy.deepcopy(f)
        #fc -= np.average(fc) it was found that DC component is important in square matrix theory, so should not be removed.
        hann=True
        T = len(fc)
        if hann:
            hw = np.arange(T)*1.0/(T-1)
            hw = (1-np.cos(2*np.pi*hw))
        nx = np.arange(len(fc))
        ni=len(fref)
        nus=np.zeros(ni).tolist()
        u=(np.zeros(ni)*(1+0j)).tolist()
        d=(np.zeros(ni)*(1+0j)).tolist()
        d2=np.zeros(ni).tolist()
        djui=(np.zeros([ni,ni])*(1+0j)).tolist()
        expdiuj=np.zeros([ni,ni])*(1+0j) #expand d[i] by u[j]
        fcdj=(np.zeros(ni)*(1+0j)).tolist()
        scale=0.1
        for i in range(ni):
            nu0,tunesorted = interpolated_fft(fc,rng=[fref[i]-df,fref[i]+df]) #find the preliminary tune approximation using 3 points in the spectrum
            #print "i=",i, " nu0,fc,tunesorted=",nu0,tunesorted
            #if i==0: sva('junk', ['in naff', [i,nu0,fc]])
            #use scale and shif origin to realize convergence.
            nu1 =yufminscaled(nu0,fc.copy(),hw, nu0, scale)[0]  #find the tune by maximize the projection weighted by hw, 
            nus[i]=nu1
            u[i]= np.exp(1j*np.pi*2*nu1*nx)
            if i==0: 
                 d[0]=u[0]
                 d2[0]=prodhw(d[0],d[0],hw) #vector product of d[0].d[0].hw, i.e. with weight hw (called window by hw)
                 expdiuj[i][i]=1 #expdiuj means expand d_i in terms of u_j
            else:
                 d[i]=u[i]
                 expdiuj[i][i]=1
                 for  k  in range(i):
                    djui[k][i]=prodhw(d[k],u[i],hw) #vector product of d[k].u[i].hw, i.e. with weight hw (called window by hw)
                    #d[i]=d[i]-  djui[j][i]  *d[j]  /d2[j]
                 d[i]=u[i]-sum(djui[k][i]  *d[k]  /d2[k] for k in range(0,i)) #Gramm-Schmidt orthogonalization step to produce the new basis vector d_i
                 expdiuj[i]=expdiuj[i]-sum( djui[k][i]*expdiuj[k]/d2[k]   for k in range(i))
                 d2[i]=prodhw(d[i],d[i], hw) #d2[i] is the norm of d_i with weight hw
            fcdj[i]=prodhw(d[i],fc, hw) #fcdj is projection with weight hw on the residual fc of d_i which is not normalized by sqrt(d2_i) yet.
            #print "i=",i, "fcdj[i]=", fcdj[i], " fcdj[i]/d2[i]=",fcdj[i]/d2[i], "d2[i]=",d2[i]
            fc=fc-d[i]*fcdj[i]/d2[i]  # d[i]/dqrt(d2[i]) is the d_i normalized according to weight hw
        fexpuj=sum(fcdj[i]*expdiuj[i]/d2[i] for i in range(ni))
        '''
        The algorithm:
        we have set of vector ui, use Gran-Smith orthogonalization to construct di, where <f|g>=Sum( f[i]*g[i]*hw[i], i=0,N-1)/N
        start from d0=u0
        d1=u1 - <d0|u1> d0 /<d0|d0>
        d2=u2 - <d0|u2> d0 /<d0|d0> - <d1|u2> d1 /<d1|d1>
        ...
        di=ui- Sum(<dk|ui> dk /<dk|dk>, k=0,i-1)
        
        Once the set of di is constructed, f is decomposed as if there are ni frequencies
        then fi is constricted as
        fi=f-Sum( <dk| f> dk, k=0,i)

        while each fi is used to find the next peak frequency nui.

        But di itself is a sum of terms with different frequencies nuk, so
        the d0,d1,d2,.. expression is used iteratively to expand di as a sum of ui

        then 
            f=Sum( <dk| f> dk, k=0,ni-1)
        and these expansion of di in terms of ui, are used to expand f as sum of ui.
        tmp=(-1)*djui[0][1]/d2[0]
        tmp1=fcdj[1] * (u[1]+tmp*u[0])
        tmp0=fcdj[0]u[0]
        so, the coefficient of u[0] is 
        fcdj[0]+fcdj[1] *tmp
        From above, we know that
        f=sum(fcdj[j]*d[j]/d2[j], for j in range(ni))
        d[i]=u[i]-sum(djui[k][i]  *d[k]  /d2[k] for k in range(i))

        Follow LascarFMA.pdf

        So, use Einstein convention,

        starting from d[0]=u[0], with i=1 as 2nd step,

        d[1]=u[1]- djui[0][1]/d2[0] *d[0] = u[1]- djui[0][1]/d2[0] *u[0] 

        so the coefficients of d[1] in terms of u[0],u[1] is

        expdiuj[1]=[0,1,0,0]- [ djui[0][1]/d2[0] ,0, 0, 0]=[ -djui[0][1]/d2[0] ,1, 0, 0]

        d[2]=u[2]- djui[k][2]/d2[k] *d[k]  

        expdiuj[2]=[0,0,1,0]- djui[k][2]/d2[k] expdiuj[k]    , with k=0,1

        d[i]=u[i]- djui[k][i]/d2[k] *d[k]  

        f=fcdj[j]/d2[j]* d[j]=fcdj[j]/d2[j] u[j]-  fcdj[j]/d2[j]*djui[k][j]  *d[k]  /d2[k] 
        '''
        return nus, fexpuj  



def naff(f,rng=[0,1],ni=10):
        '''
    NAFF: J Laskar, Physica D 56 (1992) 253-269,LascarFMA.pdf
    f: tbt data
    ni: total number of Gramm-Schmidt orthogonalizations
    returns:
    nfnu: tune peaks
    amp: absolute amplitudes
    nran: to avoid local min, try nran times random initial phase
    '''
        fc = copy.deepcopy(f)
        #fc -= np.average(fc) it was found that DC component is important in square matrix theory, so should not be removed.
        hann=True
        T = len(fc)
        if hann:
            hw = np.arange(T)*1.0/(T-1)
            hw = (1-np.cos(2*np.pi*hw))
        nx = np.arange(len(fc))
        nus=np.zeros(ni).tolist()
        u=(np.zeros(ni)*(1+0j)).tolist()
        d=(np.zeros(ni)*(1+0j)).tolist()
        d2=np.zeros(ni).tolist()
        djui=(np.zeros([ni,ni])*(1+0j)).tolist()
        expdiuj=np.zeros([ni,ni])*(1+0j) #expand d[i] by u[j]
        fcdj=(np.zeros(ni)*(1+0j)).tolist()
        scale=0.1
        for i in range(ni):
            nu0,tunesorted = interpolated_fft(fc,rng=rng,ith=1,hann=True) #find the preliminary tune approximation using 3 points in the spectrum
            #print "i=",i, " nu0,fc,tunesorted=",nu0,tunesorted
            #if i==0: sva('junk', ['in naff', [i,nu0,fc]])
            #use scale and shif origin to realize convergence.
            nu1 =yufminscaled(nu0,fc.copy(),hw, nu0, scale)[0]  #find the tune by maximize the projection weighted by hw, 
            nus[i]=nu1
            u[i]= np.exp(1j*np.pi*2*nu1*nx)
            if i==0: 
                 d[0]=u[0]
                 d2[0]=prodhw(d[0],d[0],hw) #vector product of d[0].d[0].hw, i.e. with weight hw (called window by hw)
                 expdiuj[i][i]=1 #expdiuj means expand d_i in terms of u_j
            else:
                 d[i]=u[i]
                 expdiuj[i][i]=1
                 for  k  in range(i):
                    djui[k][i]=prodhw(d[k],u[i],hw) #vector product of d[k].u[i].hw, i.e. with weight hw (called window by hw)
                    #d[i]=d[i]-  djui[j][i]  *d[j]  /d2[j]
                 d[i]=u[i]-sum(djui[k][i]  *d[k]  /d2[k] for k in range(0,i)) #Gramm-Schmidt orthogonalization step to produce the new basis vector d_i
                 expdiuj[i]=expdiuj[i]-sum( djui[k][i]*expdiuj[k]/d2[k]   for k in range(i))
                 d2[i]=prodhw(d[i],d[i], hw) #d2[i] is the norm of d_i with weight hw
            fcdj[i]=prodhw(d[i],fc, hw) #fcdj is projection with weight hw on the residual fc of d_i which is not normalized by sqrt(d2_i) yet.
            #print "i=",i, "fcdj[i]=", fcdj[i], " fcdj[i]/d2[i]=",fcdj[i]/d2[i], "d2[i]=",d2[i]
            fc=fc-d[i]*fcdj[i]/d2[i]  # d[i]/dqrt(d2[i]) is the d_i normalized according to weight hw
        fexpuj=sum(fcdj[i]*expdiuj[i]/d2[i] for i in range(ni))
        '''
        The algorithm:
        we have set of vector ui, use Gran-Smith orthogonalization to construct di, where <f|g>=Sum( f[i]*g[i]*hw[i], i=0,N-1)/N
        start from d0=u0
        d1=u1 - <d0|u1> d0 /<d0|d0>
        d2=u2 - <d0|u2> d0 /<d0|d0> - <d1|u2> d1 /<d1|d1>
        ...
        di=ui- Sum(<dk|ui> dk /<dk|dk>, k=0,i-1)
        
        Once the set of di is constructed, f is decomposed as if there are ni frequencies
        then fi is constricted as
        fi=f-Sum( <dk| f> dk, k=0,i)

        while each fi is used to find the next peak frequency nui.

        But di itself is a sum of terms with different frequencies nuk, so
        the d0,d1,d2,.. expression is used iteratively to expand di as a sum of ui

        then 
            f=Sum( <dk| f> dk, k=0,ni-1)
        and these expansion of di in terms of ui, are used to expand f as sum of ui.
        tmp=(-1)*djui[0][1]/d2[0]
        tmp1=fcdj[1] * (u[1]+tmp*u[0])
        tmp0=fcdj[0]u[0]
        so, the coefficient of u[0] is 
        fcdj[0]+fcdj[1] *tmp
        From above, we know that
        f=sum(fcdj[j]*d[j]/d2[j], for j in range(ni))
        d[i]=u[i]-sum(djui[k][i]  *d[k]  /d2[k] for k in range(i))

        Follow LascarFMA.pdf

        So, use Einstein convention,

        starting from d[0]=u[0], with i=1 as 2nd step,

        d[1]=u[1]- djui[0][1]/d2[0] *d[0] = u[1]- djui[0][1]/d2[0] *u[0] 

        so the coefficients of d[1] in terms of u[0],u[1] is

        expdiuj[1]=[0,1,0,0]- [ djui[0][1]/d2[0] ,0, 0, 0]=[ -djui[0][1]/d2[0] ,1, 0, 0]

        d[2]=u[2]- djui[k][2]/d2[k] *d[k]  

        expdiuj[2]=[0,0,1,0]- djui[k][2]/d2[k] expdiuj[k]    , with k=0,1

        d[i]=u[i]- djui[k][i]/d2[k] *d[k]  

        f=fcdj[j]/d2[j]* d[j]=fcdj[j]/d2[j] u[j]-  fcdj[j]/d2[j]*djui[k][j]  *d[k]  /d2[k] 
        '''
        return nus, fexpuj  


'''
The following is copied from naff.f90, but the Grand-Smith basis were not nomalzed in naff.f90, so the code is likely wrong!
    u(i,:) = ed(freqs(i),size_data) #according to projection, it is divided by T after sum, so it is normalized already.
    do j=2,i
        u(i,:) = u(i,:) - projdd(u(j-1,:),u(i,:))*u(j-1,:)
    enddo
    amps(i) = projdd(u(i,:),cdata)
    cdata = cdata - amps(i)*u(i,:)
    enddo
'''

# --- END of NAFF
'''
About  fft
    fft of xn

    Xk=Sigma n=0N--1 xn.e(--i 2 pi  n k/ N)
    so  
    Xk=Sigma n=0N-1  (  exp(2j pi nu n).e(- 2j pi k/N n)  )=Sigma n=0N--1  (  exp(2j pi (nu-k/N) n)  )  =(1- exp(2j pi (nu-k/N) N)))/(1- exp(2j pi (nu-k/N)))

    =exp(j pi (nu-k/N) (N-1))*sin( pi  (nu-k/N)N)/sin( pi (nu-k/N))
    The following shows that this expression agrees with fft of its spectrum:

    wy10=array([np.exp(1j*np.pi*(nuswy1[0]-k/N)*(N-1))*np.sin(np.pi*(nuswy1[0]-k/N)*N)/np.sin(np.pi*(nuswy1[0]-k/N)) for k in range(1200)])
    plt.plot(ff, wy10.real,'r-')
    plt.plot(ff,fuk[0].real,'b-')
    plt.axis([0.0,0.1,-4.e2,8.e2])
'''


def projectionhw(nu,*args):
    '''
    give nu, phase, check the different between
    f: TbT data
    n: [nux,phi]
    '''
    v,hw=copy.deepcopy(args)
    i = np.arange(len(v))
    v1 = np.exp(-np.pi*2*nu*i*1j)
    v *= hw #v=>v*hw
    a =-abs( np.dot(v,v1)/len(v))*abs( np.dot(v,v1)/len(v))
    return a

def projectionhwscale(x,*args):
    fcc,hw,nu0,scale=copy.deepcopy(args)
    nux=nu0+x*scale  #change variable so that nux varies around nus0
    #print "x=",x, " nux=", nux, "projectionhw(nux,fcc,hw)=",projectionhw(nux,fcc,hw)
    return projectionhw(nux,fcc,hw)

def yufminscaled(nu, *args):
    fcc,hw,nu0, scale=copy.deepcopy(args)
    x=(nu-nu0)/scale
    sol=opt.fmin(projectionhwscale, x, args=(fcc,hw,nu0, scale),xtol=1e-5,ftol=1e-5,disp=0, callback=None, full_output=True)
    return nu0+sol[0]*scale


def prodhw(u1, u2, hw):
    u1c=copy.deepcopy(u1)
    u1c *= hw #u1=> u1*hw
    return np.dot(np.conj(u1c),u2)/len(u1c)

'''
study scaling fmin


scale=0.1
def fscale(x,*args):
    fcc,hw,nu0,scale=args
    nux=nu0+x*scale  #change variable so that nux varies around nus0
    print "x=",x, " nux=", nux, "projectionhw(nux,fcc,hw)=",projectionhw(nux,fcc,hw)
    return projectionhw(nux,fcc,hw)

def callb(x):
    tmp=fscale(x, fc.copy(),hw,nu0, scale)
    nux=nu0+x*scale  #change variable so that nux varies around nus0
    print "nux=",nux[0], " x=",x," tmp=",tmp
    return 


tmp3=opt.fmin(fscale, 0.1, args=(fc.copy(),hw,nu0, scale),xtol=1e-3,ftol=1e-3,disp=True, callback=callb, full_output=True)
print tmp3
'''
'''
naffpeaks uses hanning projection, haff use window=1 projection
Compare naffpeaks with naff in yunaff:
naffpeaks get error of 5.2e-5 for the first term , 1.18e-6 for 2nd term
naff      get error of 2e-4   for the first term , 1e-4    for 2nd term

In [280]: %run yunaff.py

Check coefficietes of u[0].

fexpuj[0]-np.exp(0.12*1j)= (6.26073980714e-06-5.20851031719e-05j)

Check coefficietes of u[1].

fexpuj[1]-np.exp(-0.37*1j)*0.1= (4.63822318178e-07+1.18643549786e-06j)
nus= [[0.22000013148659364], [0.73999996810034585]]

In [281]: %run FortNAFF-master/testyunaff.py

Check coefficietes of u[0].

fcdj[0]+fcdj[1] *tmp-np.exp(0.12*1j)= (-1.5483631144e-05-0.000195842143278j)
fcdj[0]-np.exp(0.12*1j)= (0.000729949619145-0.0003819831274j)
fexpuj[0]-np.exp(0.12*1j)= (-1.55276894049e-05-0.000195831141558j)

Check coefficietes of u[1].

fcdj[1]-np.exp(-0.37*1j)*0.1= (-0.000106592811173-0.000113476529399j)
fexpuj[1]-np.exp(-0.37*1j)*0.1= (-0.000101088661936-0.000115620536639j)
nus= [0.22000048679858392, 0.7400031844539825]
'''

#test yunaff.py:
'''
ztest =[np.exp(2*n*np.pi*.22*1j+0.12*1j)+np.exp(-2*n*np.pi*.26*1j-0.37*1j)*0.1 for n in range(128)]

nus, fexpuj=naff(ztest,ni=2)
print "\nCheck coefficietes of u[0].\n"
print "fexpuj[0]-np.exp(0.12*1j)=",fexpuj[0]-np.exp(0.12*1j)

print "\nCheck coefficietes of u[1].\n"
print "fexpuj[1]-np.exp(-0.37*1j)*0.1=",fexpuj[1]-np.exp(-0.37*1j)*0.1
print "nus=",nus
'''
