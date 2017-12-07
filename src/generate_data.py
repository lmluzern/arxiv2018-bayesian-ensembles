'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

from data.load_data import load_crowdsourcing_data, load_argmin_data
from algorithm import bac
from baselines import majority_voting
import numpy as np

# load data
annos, doc_start = load_crowdsourcing_data()
N = float(annos.shape[0])
K = annos.shape[1]
L = 3

# run majority voting
base = majority_voting.MajorityVoting(annos, 3)
maj, votes = base.vote()

# run BAC 
bac_ = bac.BAC(L=L, K=K, nu0=np.ones((L+1, L)) * 0.1, alpha0=0.1 * np.ones((L, L, L+1, K)) + 0.1 * np.eye(3)[:, :, None, None])
probs, agg = bac_.run(annos, doc_start)

np.savetxt('./data/crowdsourcing/gen/probs2.csv', probs, delimiter=',')
np.savetxt('./data/crowdsourcing/gen/agg2.csv', agg, delimiter=',')

np.savetxt('./data/crowdsourcing/gen/majority2.csv', maj , delimiter=',')
np.savetxt('./data/crowdsourcing/gen/votes2.csv', votes, delimiter=',')

# load data
# probs = np.genfromtxt('./data/crowdsourcing/gen/probs.csv', delimiter=',')
# agg = np.genfromtxt('./data/crowdsourcing/gen/agg.csv', delimiter=',')

# build gold standard
gold = -np.ones_like(agg)

for i in xrange(agg.shape[0]):
    if ((agg[i] == int(maj[i])) and (i, probs[int(agg[i])] > 0.5)):
        gold[i] = agg[i]
        
print "Generated the gold data for testing on the crowdsourcing dataset. There are %i gold-labelled and %i unconfirmed data points." % (np.sum(gold!=-1), np.sum(gold==-1))    
    
np.savetxt('./data/crowdsourcing/gen/gold2.csv', gold, fmt='%s', delimiter=',')

if __name__ == '__main__':
    pass
