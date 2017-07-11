'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

from src.baselines import ibcc, clustering, majority_voting
from src.algorithm import bac
from src.data import data_utils
import ConfigParser
import os, subprocess
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
import metrics

class Experiment(object):
    '''
    classdocs
    '''
    
    generator = None
    
    config_file = None
    
    param_values = None
    param_idx = None
    param_fixed = None
    
    acc_bias = None
    miss_bias = None
    short_bias = None
    
    param_names = ['acc_bias', 'miss_bias', 'short_bias', 'num_docs', 'doc_length', 'group_sizes']
    score_names = ['accuracy', 'precision', 'recall', 'f1-score', 'auc-score', 'cross-entropy-error', 'count error', 'number of invalid labels', 'mean length error']
    
    methods = None
    
    num_runs = None
    doc_length = None
    group_sizes = None
    num_docs = None
    
    num_classes = None
    
    output_dir = '/output/'
    
    save_results = False
    save_plots = False
    show_plots = False
    
    postprocess = True
    
    '''
    Constructor
    '''
    def __init__(self, generator, config=None):
        '''
        Constructor
        '''
        self.generator = generator
        
        if not (config == None):
            self.config_file = config
            self.read_config_file()
            
            
    def read_config_file(self):
        
        print 'Reading experiment config file...'
        
        parser = ConfigParser.ConfigParser()
        parser.read(self.config_file)
    
        # set up parameters
        parameters = dict(parser.items('parameters'))
        self.param_idx = int(parameters['idx'].split('#')[0].strip())
        self.param_values = np.array(eval(parameters['values'].split('#')[0].strip()))
        
        self.acc_bias = np.array(eval(parameters['acc_bias'].split('#')[0].strip()))
        self.miss_bias = np.array(eval(parameters['miss_bias'].split('#')[0].strip()))
        self.short_bias = np.array(eval(parameters['short_bias'].split('#')[0].strip()))
        
        
        self.methods = eval(parameters['methods'].split('#')[0].strip())
        
        self.postprocess = eval(parameters['postprocess'].split('#')[0].strip())
        
        print self.methods
                
        self.num_runs = int(parameters['num_runs'].split('#')[0].strip())
        self.doc_length = int(parameters['doc_length'].split('#')[0].strip())
        self.group_sizes = np.array(eval(parameters['group_sizes'].split('#')[0].strip()))
        self.num_classes = int(parameters['num_classes'].split('#')[0].strip())
        self.num_docs = int(parameters['num_docs'].split('#')[0].strip())
        
        # set up output
        parameters = dict(parser.items('output'))
        self.output_dir = parameters['output_dir']
        self.save_results = eval(parameters['save_results'].split('#')[0].strip())
        self.save_plots = eval(parameters['save_plots'].split('#')[0].strip())
        self.show_plots = eval(parameters['show_plots'].split('#')[0].strip())
        
    
    def run_methods(self, annotations, ground_truth, doc_start, param_idx):
        for method_idx in xrange(len(self.methods)):
            
            if self.methods[method_idx] == 'majority':
                    
                mv = majority_voting.MajorityVoting(annotations, self.num_classes)
                agg, probs = mv.vote()
            
            if self.methods[method_idx] == 'clustering':
                    
                cl = clustering.Clustering(ground_truth, annotations)
                agg = cl.run()
                    
                probs = np.zeros((ground_truth.shape[0], self.num_classes))
                for k in xrange(ground_truth.shape[0]):
                    probs[k, int(agg[k])] = 1      
                
            if self.methods[method_idx] == 'mace':
                subprocess.call(['java', '-jar', 'MACE/MACE.jar', '--distribution', '--prefix', 'output/data/mace', 'output/data/annotations.csv'])
                
                result = np.genfromtxt('output/data/mace.prediction')
                    
                agg = result[:, 0]
                    
                probs = np.zeros((ground_truth.shape[0], self.num_classes))
                for i in xrange(result.shape[0]):
                    for j in xrange(0, self.num_classes * 2, 2):
                        probs[i, int(result[i, j])] = result[i, j + 1]  
                
            if self.methods[method_idx] == 'ibcc':
                
                alpha0 = np.ones((self.num_classes, self.num_classes, 10))
                alpha0[:, :, 5] = 2.0
                alpha0[np.arange(3), np.arange(3), :] += 1.0
                
                nu0 = np.array([1, 1, 1], dtype=float)
                
                ibc = ibcc.IBCC(nclasses=3, nscores=3, nu0=nu0, alpha0=alpha0)
                probs = ibc.combine_classifications(annotations, table_format=True)
                
                agg = probs.argmax(axis=1)
                
            if self.methods[method_idx] == 'bac':
                
                alg = bac.BAC(L=3, K=annotations.shape[1])
                
                probs = alg.run(annotations, doc_start)
                agg = probs.argmax(axis=1)
                    
            #self.update_scores(scores, param_idx, method_idx, agg, ground_truth, probs, doc_start)
        
        
    def single_run(self):
        
        scores = np.zeros((self.param_values.shape[0], len(self.score_names), len(self.methods)))
       
        for param_idx in xrange(self.param_values.shape[0]):
            # update tested parameter
            if self.param_idx == 0:
                self.acc_bias = self.param_values[param_idx]
            elif self.param_idx == 1:
                self.miss_bias = self.param_values[param_idx]
            elif self.param_idx == 2:
                self.short_bias = self.param_values[param_idx]
            elif self.param_idx == 3:
                self.num_docs = self.param_values[param_idx]
            elif self.param_idx == 4:
                self.doc_length = self.param_values[param_idx]
            elif self.param_idx == 5:
                self.group_sizes = self.param_values[param_idx]
            else:
                print 'Encountered invalid test index!'
                
            self.generator.init_crowd_models(self.acc_bias, self.miss_bias, self.short_bias, self.group_sizes)
            ground_truth, annotations, doc_start = self.generator.generate_dataset(num_docs=self.num_docs, doc_length=self.doc_length, group_sizes=self.group_sizes, save_to_file=True)
            agg = None
            
            for method_idx in xrange(len(self.methods)):
            
                if self.methods[method_idx] == 'majority':
                    
                    mv = majority_voting.MajorityVoting(annotations, self.num_classes)
                    agg, probs = mv.vote()
            
                if self.methods[method_idx] == 'clustering':
                    
                    cl = clustering.Clustering(ground_truth, annotations)
                    agg = cl.run()
                    
                    probs = np.zeros((ground_truth.shape[0], self.num_classes))
                    for k in xrange(ground_truth.shape[0]):
                        probs[k, int(agg[k])] = 1      
                
                if self.methods[method_idx] == 'mace':
                    subprocess.call(['java', '-jar', 'MACE/MACE.jar', '--distribution', '--prefix', 'output/data/mace', 'output/data/annotations.csv'])
                
                    result = np.genfromtxt('output/data/mace.prediction')
                    
                    agg = result[:, 0]
                    
                    probs = np.zeros((ground_truth.shape[0], self.num_classes))
                    for i in xrange(result.shape[0]):
                        for j in xrange(0, self.num_classes * 2, 2):
                            probs[i, int(result[i, j])] = result[i, j + 1]  
                
                if self.methods[method_idx] == 'ibcc':
                
                    alpha0 = np.ones((self.num_classes, self.num_classes, 10))
                    alpha0[:, :, 5] = 2.0
                    alpha0[np.arange(3), np.arange(3), :] += 1.0
                
                    nu0 = np.array([1, 1, 1], dtype=float)
                
                    ibc = ibcc.IBCC(nclasses=3, nscores=3, nu0=nu0, alpha0=alpha0)
                    probs = ibc.combine_classifications(annotations, table_format=True)
                
                    agg = probs.argmax(axis=1)
                
                if self.methods[method_idx] == 'bac':
                
                    alg = bac.BAC(L=3, K=annotations.shape[1])
                
                    probs = alg.run(annotations, doc_start)
                
                    agg = probs.argmax(axis=1)
                    
                self.update_scores(scores, param_idx, method_idx, agg, ground_truth, probs, doc_start)
        
        return scores
    
    
    def update_scores(self, scores, param_idx, method_idx, agg, ground_truth, probs, doc_start):
        
        scores[param_idx, 7, method_idx] = metrics.num_invalid_labels(agg)
        
        if self.postprocess:
            agg = data_utils.postprocess(agg, doc_start)
        
        scores[param_idx, 0, method_idx] = skm.accuracy_score(ground_truth[:, 1], agg)
        scores[param_idx, 1, method_idx], scores[param_idx, 2, method_idx], scores[param_idx, 3, method_idx], _ = skm.precision_recall_fscore_support(ground_truth[:, 1], agg, pos_label=0, average='macro')
        auc_score = skm.roc_auc_score(ground_truth[:, 1] == 0, agg == 0) * np.sum(ground_truth[:, 1] == 0)
        auc_score += skm.roc_auc_score(ground_truth[:, 1] == 1, agg == 1) * np.sum(ground_truth[:, 1] == 1)
        auc_score += skm.roc_auc_score(ground_truth[:, 1] == 2, agg == 2) * np.sum(ground_truth[:, 1] == 2)
        scores[param_idx, 4, method_idx] = auc_score / float(ground_truth.shape[0])
        scores[param_idx, 5, method_idx] = skm.log_loss(ground_truth[:, 1], probs, eps=1e-100)
        scores[param_idx, 6, method_idx] = metrics.abs_count_error(agg, ground_truth[:, 1])
        
        scores[param_idx, 8, method_idx] = metrics.mean_length_error(agg, ground_truth[:, 1])
    
    
    def run(self):
        
        results = np.zeros((self.param_values.shape[0], len(self.score_names), len(self.methods), self.num_runs))
        
        print 'Running experiments...'
        
        for i in xrange(self.num_runs):   
            print 'Run number:', i 
            results[:, :, :, i] = self.single_run() 
            
        if self.save_results:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            print 'Saving results...'
            np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])

        if self.show_plots or self.save_plots:
            self.plot_results(results, self.show_plots, self.save_plots, self.output_dir)
        
        return results
    
    
    def run_config(self):
        if self.config_file == None:
            raise RuntimeError('No config file specified.')
        
        return self.run()
    
    
    def plot_results(self, results, show_plot=False, save_plot=False, output_dir='/output/'):
        
        # create output directory if necessary
        if save_plot and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # initialise values for x axis
        if (self.param_values.ndim > 1):
            x_vals = self.param_values[:, 0]
        else:
            x_vals = self.param_values
            
        # initialise x-tick labels 
        x_ticks_labels = map(str, self.param_values)
            
        for i in xrange(len(self.score_names)):    
            for j in xrange(len(self.methods)):
                
                plt.errorbar(x_vals, np.mean(results[:, i, j, :], 1), yerr=np.std(results[:, i, j, :], 1), label=self.methods[j])
                
            
            plt.legend(loc=0)
        
            plt.title('parameter influence')
            plt.ylabel(self.score_names[i])
            plt.xlabel(self.param_names[self.param_idx])
            plt.xticks(x_vals, x_ticks_labels)
            plt.ylim([0, np.max([1, np.max(results[:, i, :, :])])])
        
            if save_plot:
                print 'Saving plot...'
                plt.savefig(self.output_dir + 'plot_' + self.score_names[i] + '.png') 
                plt.clf()
        
            if show_plot:
                plt.show()
