#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:09:37 2018
Implementation of HOPITOR Webserver
by Abdul Hannan Basit
as part of  Abdul Hannan Basit, Wajid Abbasi, Amina Asif, Sadaf Gull, and Fayyaz ul Amir Afsar Minhas- "Training host-pathogen protein-protein interaction predictors" (pre-print release).
@author: cuser
"""

import numpy as np
from sklearn.externals import joblib    
from itertools import product #to create dictionary 
  

class kmerFE:
    """
    Feature extraction class
    """
    def __init__(self,kk=3):                    
        self.kk = kk
        self.dic={'A':'1','G':'1','V':'1','I':'2','L':'2','F':'2','P':'2',
        'Y':'3','M':'3','T':'3','S':'3','H':'4','N':'4','Q':'4','W':'4',
        'R':'5','K':'5','D':'6','E':'6','C':'7'}
        keywords = [''.join(i) for i in product("1234567", repeat = kk)]
        self.idict=dict(zip(keywords,range(len(keywords))))
        
    def encode(self,text): #to convert amino acids to group
        for i, j in self.dic.iteritems():
            text = text.replace(i, j)
        return text 

    def feature_kmer(self,read):    #returns k-mer dictionary with counts in seq read
        read  = self.encode(str(read))
        num_kmers = len(read) - self.kk + 1
        Z = np.zeros(len(self.idict))
        for i in range(num_kmers):
            # Slice the string to get the kmer
            kmer = read[i:i+self.kk]
            if kmer in self.idict:
                Z[self.idict[kmer]]+=1
        x1 = ([p for p in Z] -min(Z))/(max(Z)-min(Z))
        return x1
    
    def pair(self,host,viral):
        x1 = self.feature_kmer(host)
        x2 = self.feature_kmer(viral)
        X = np.hstack((x1,x2))
        return X.reshape(1,-1)    
      
def compute_score(fe,clf,host,viral):         
    X=fe.pair(host,viral)    
    prob=clf.predict_proba(X)[0][1]
    pred=clf.predict(X)[0]
    return prob,pred

if __name__ == "__main__":
    fe = kmerFE()
    clf=joblib.load('xg-final.pkl')   
    
    viral='MAEEQARHVKNGLECIRALKAEPIGSLAIEEAMAAWSEISDNPGQERATCREEKAGSSGLSKPCLSAIGSTEGGAPRIRGQGPGESDDDAETLGIPPRNLQASSTGLQCYYVYDHSGEAVKGIQDADSIMVQSGLDGDSTLSGGDNESENSDVDIGEPDTEGYAITDRGSAPISMGFRASDVETAEGGEIHELLRLQSRGNNFPKLGKTLNVPPPPDPGRASTSGTPIKKGHRREISLIWNGDRVFIDRWCNPMCSKVTLGTIRARCTCGECPRVCEQCRTDTGVDTRIWYHNLPEIPE'
    host='MTANRDAALSSHRHPGCAQRPRTPTFASSSQRRSAFGFDDGNFPGLGERSHAPGSRLGARRRAKTARGLRGHRQRGAGAGLSRPGSARAPSPPRPGGPENPGGVLSVELPGLLAQLARSFALLLPVYALGYLGLSFSWVLLALALLAWCRRSRGLKALRLCRALALLEDEERVVRLGVRACDLPAWVHFPDTERAEWLNKTVKHMWPFICQFIEKLFRETIEPAVRGANTHLSTFSFTKVDVGQQPLRINGVKVYTENVDKRQIILDLQISFVGNCEIDLEIKRYFCRAGVKSIQIHGTMRVILEPLIGDMPLVGALSIFFLRKPLLEINWTGLTNLLDVPGLNGLSDTIILDIISNYLVLPNRITVPLVSEVQIAQLRFPVPKGVLRIHFIEAQDLQGKDTYLKGLVKGKSDPYGIIRVGNQIFQSRVIKENLSPKWNEVYEALVYEHPGQELEIELFDEDPDKDDFLGSLMIDLIEVEKERLLDEWFTLDEVPKGKLHLRLEWLTLMPNASNLDKVLTDIKADKDQANDGLSSALLILYLDSARNLPSGKKISSNPNPVVQMSVGHKAQESKIRYKTNEPVWEENFTFFIHNPKRQDLEVEVRDEQHQCSLGNLKVPLSQLLTSEDMTVSQRFQLSNSGPNSTIKMKIALRVLHLEKRERPPDHQHSAQVKRPSVSKEGRKTSIKSHMSGSPGPGGSNTAPSTPVIGGSDKPGMEEKAQPPEAGPQGLHDLGRSSSSLLASPGHISVKEPTPSIASDISLPIATQELRQRLRQLENGTTLGQSPLGQIQLTIRHSSQRNKLIVVVHACRNLIAFSEDGSDPYVRMYLLPDKRRSGRRKTHVSKKTLNPVFDQSFDFSVSLPEVQRRTLDVAVKNSGGFLSKDKGLLGKVLVALASEELAKGWTQWYDLTEDGTRPQAMT'
    
    prob,pred = compute_score(fe,clf,host,viral)
    print "Probability to Interact=",prob,"Interaction Prediction=",pred

            
            
    