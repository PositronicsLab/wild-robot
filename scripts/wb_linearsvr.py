"""
args
  rbf
  sim[ode, dart]
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 
import sys
import time
import json
#import pickle

#------------------------------------------------------------------------------
"""
"""
def sec_to_str(t):
  return "%d:%02d:%02d.%03d" % reduce(lambda ll,b : divmod(ll[0],b) + ll[1:], [(t*1000,),1000,60,60])

#------------------------------------------------------------------------------
"""
Generates and returns a timestamp
"""
def timestamp():
  return time.time()

#------------------------------------------------------------------------------
"""
Generates a timestamp and computes the difference between that timestamp and
a previous timestamp.  Prints the difference
"""
def log_time(t0, msg):
  t1 = timestamp()
  dt = t1-t0
  s = sec_to_str(dt)
  print 'time to ' + msg + ': ' + s

#------------------------------------------------------------------------------
"""
Computes an initial timestamp and returns it
"""
def log_start():
  start_time = timestamp()
  log = {'start_time' : start_time}
  return log

#------------------------------------------------------------------------------
"""
Computes an end timestamp and computes the difference between the initial log
timestamp.  Prints the difference.
"""
def log_stop(log):
  start_time = log.get('start_time')
  end_time = timestamp()
  delta_t = end_time - start_time;
  s = sec_to_str(delta_t)
  print 'execution time: ' + s

#------------------------------------------------------------------------------
"""
Originally logged directly to a file.  Now just prints to the console as logging
has been supplanted by piping stdout to a file so that there is no need for
special handling of data when logging.
"""
def log_write(s):
  print s

#------------------------------------------------------------------------------
"""
Strips the position fields out of a data set.  This was done during early 
evaluation of regression.  Intuitively, it seems position might not contribute
meaningful information to the regression; however, this is not the case.
This function is turned off, but references to it remain commented out.
"""
def remove_position(x):
  #print x.shape
  a = np.reshape(x, (-1,15))
  a = a[:,3:15]
  #print a.shape
  return np.reshape(a, (-1,24))
  
#------------------------------------------------------------------------------
"""
Load a data from disk as a numpy array
"""
def load_data_file( fname ):
  X = np.loadtxt(fname) 
  #X = remove_position(np.loadtxt(fname)) 
  return X

#------------------------------------------------------------------------------
#"""
#"""
#def byteify(input):
#  if isinstance(input, dict):
#    return {byteify(key): byteify(value)
#      for key, value in input.iteritems()}
#  elif isinstance(input, list):
#    return [byteify(element) for element in input]
#  elif isinstance(input, unicode):
#    return input.encode('utf-8')
#  else:
#    return input

#------------------------------------------------------------------------------
"""
Save a python dictionary to disk in json format
"""
def save_dictionary( fname, dictionary ):
  with open(fname, 'w') as f:
    f.write(json.dumps(dictionary))
    f.close()

#------------------------------------------------------------------------------
"""
Load a json formatted dictionary from disk as a python dictionary
"""
def load_dictionary( fname ):
  with open(fname,'r') as f:
    #d = byteify(json.loads(file.read()))
    d = json.loads(f.read())
    f.close()
  return d

#------------------------------------------------------------------------------
"""
Pickles an object and saves it to disk.  Unfortunately, joblib does not create
a platform independent format, so the object may not be usable across multiple 
machines.
"""
def save_obj( fname, obj ):
#  with open(fname + '.pkl', 'wb') as f:
#    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#    f.close()
  joblib.dump(obj, fname+'.pkl')

#------------------------------------------------------------------------------
"""
Loads a pickled object from disk
"""
def load_obj( fname ):
#  with open(fname + '.pkl', 'rb') as f:
#    p = pickle.load(f)
#    f.close()
#  return p 
  return joblib.load(fname+'.pkl')
#------------------------------------------------------------------------------
"""
Load all the testset data as a single numpy array
"""
def load_data_set():
  X01 = load_data_file('trial_01_testset.txt')
  X02 = load_data_file('trial_02_testset.txt')
  X03 = load_data_file('trial_03_testset.txt')
  X04 = load_data_file('trial_04_testset.txt')
  X05 = load_data_file('trial_05_testset.txt')
  X06 = load_data_file('trial_06_testset.txt')
  X07 = load_data_file('trial_07_testset.txt')
  X08 = load_data_file('trial_08_testset.txt')
  X09 = load_data_file('trial_09_testset.txt')
  X10 = load_data_file('trial_10_testset.txt')

  X = np.concatenate((X01,X02,X03,X04,X05,X06,X07,X08,X09,X10),axis=0)

  return X

#------------------------------------------------------------------------------
"""
Load training data as numpy arrays.
"""
def load_training_file( fname, yval ):
  X = np.loadtxt(fname) 
  #X = remove_position(np.loadtxt(fname)) 
  y = np.empty([len(X)])
  y.fill(yval)
  return X, y

#------------------------------------------------------------------------------
"""
Load all the simulation data as a single numpy array
"""
def load_simulation_set( args ):
  sim = args.get('simulator')
  offsets = args.get('offsets')

  print sim
  print offsets

  Xm4, ym4 = load_training_file(sim+'-offset-4_trainingset.txt', offsets[0])
  Xm3, ym3 = load_training_file(sim+'-offset-3_trainingset.txt', offsets[1])
  Xm2, ym2 = load_training_file(sim+'-offset-2_trainingset.txt', offsets[2])
  Xm1, ym1 = load_training_file(sim+'-offset-1_trainingset.txt', offsets[3])
  Xm0, ym0 = load_training_file(sim+'-offset-0_trainingset.txt', offsets[4])
  Xp1, yp1 = load_training_file(sim+'-offset+1_trainingset.txt', offsets[5])
  Xp2, yp2 = load_training_file(sim+'-offset+2_trainingset.txt', offsets[6])
  Xp3, yp3 = load_training_file(sim+'-offset+3_trainingset.txt', offsets[7])
  Xp4, yp4 = load_training_file(sim+'-offset+4_trainingset.txt', offsets[8])

  X = np.concatenate((Xm4,Xm3,Xm2,Xm1,Xm0,Xp1,Xp2,Xp3,Xp4),axis=0)
  y = np.concatenate((ym4,ym3,ym2,ym1,ym0,yp1,yp2,yp3,yp4),axis=0)

  return X, y

#------------------------------------------------------------------------------
"""
Decimate training set data.  Removes any records from the arrays that does not
align on a step
"""
def decimate_training_set(X, y, step=10):
  idxs = np.arange(0,len(X),step)
  X = np.take(X, idxs, axis=0)
  y = np.take(y, idxs, axis=0)
  return X, y

#------------------------------------------------------------------------------
"""
Decimate test set data.  Removes any records from the array that does not
align on a step
"""
def decimate_test_set(X, step=10):
  idxs = np.arange(0,len(X),step)
  X = np.take(X, idxs)
  return X

#------------------------------------------------------------------------------
"""
Shuffles the data provided by the input arrays and returns new arrays that 
contain the shuffled data and the sequencing
"""
def shuffle_data( Xin, yin ):
  idxs = np.arange(len(Xin))
  np.random.shuffle(idxs)
  X = np.take(Xin, idxs, axis=0)
  y = np.take(yin, idxs, axis=0)
  return X, y, idxs

#------------------------------------------------------------------------------
"""
Save a numpy array to disk.  Ensures that the data is written with a large 
number of significant digits
"""
def save_y_data(fname, y):
  np.savetxt(fname, y, delimiter=' ', fmt='%1.15e')

#------------------------------------------------------------------------------
"""
DEPRECATED
"""
def get_y_data_filename(pfx, sfx, sim):
  return pfx+'_' + sim + '_' + sfx + '.txt'

#------------------------------------------------------------------------------
"""
Cross-validation step
"""
def svr_cv(args, X, y):
  testing = args.get('testing')
  alg = args.get('algorithm')
  cvs = args.get('cross-validations')

  scaler = preprocessing.StandardScaler()

  scores = ['precision', 'recall' ]
  score = scores[0]
  #svr = LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
  #   intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
  #   random_state=0, tol=0.0001, verbose=0)
  svr = LinearSVR(random_state=0)
  
  C = [1] 
  eps = [1e-6]
  max_iter = [1e6]
  tol = [1e-5]

  hyperparameters = { 'linearsvr__C': C, 'linearsvr__epsilon' : eps, 'linearsvr__max_iter' : max_iter, 'linearsvr__tol' : tol }
  
  pipeline = make_pipeline(scaler, svr)

  print "cross-validating"
  t0 = timestamp()
  clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, refit=True)
  log_time(t0,"initialize GridSearchCV")
  #print clf
  t0 = timestamp()
  clf.fit(X, y)
  log_time(t0,"fit to GridSearchCV")

  log_write('clf:' + str(clf))
  #log_time(t0, 'preprocess data')
  print "Best parameters found:"
  print clf.best_params_

  t0 = timestamp()
  score_train = clf.score(X, y)
  log_time(t0,"compute training score")
  #score_test = clf.score(X_test, y_test)
  #svr_params = clf.get_params()

  print 'score_train: ' + str(score_train)
  #print 'score_test: ' + str(score_test)
  #print 'svr_params: ' + str(svr_params)

  #score = cross_val_score()
  #print 'training neg_mean_sq_error:' + str(cross_val_score(clf, X, y, scoring='neg_mean_squared_error'))

  result = { 'clf': clf, 'params':clf.best_params_, 'score_train':score_train }
  return result
 
  
##------------------------------------------------------------------------------
#"""
#Training step
#"""
#def svr_train(args, X, y):
#  testing = args.get('testing')
#  alg = args.get('algorithm')
#  params = args.get('params')
#  #print params
#
#  scaler = preprocessing.StandardScaler()
#
#  scores = ['precision', 'recall' ]
#  score = scores[0]
#  #svr = LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
#  #   intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
#  #   random_state=0, tol=0.0001, verbose=0)
#  #svr = LinearSVR(random_state=0)
#
#  C=params.get(str(alg+'__C'))
#  eps=params.get(str(alg+'__epsilon'))
#  itr=params.get(str(alg+'__max_iter'))
#  tol=params.get(str(alg+'__tol'))
#
#  svr = LinearSVR(C=C, epsilon=eps, max_iter=itr, tol=tol)
#  
#  #C = [0] 
#  #eps = [1e-1]
#  #max_iter = [1e3]
#  #tol = [1e-5]
#
#  #hyperparameters = { 'linearsvr__C': C, 'linearsvr__epsilon' : eps, 'linearsvr__max_iter' : max_iter, 'linearsvr__tol' : tol }
#  
#  clf = make_pipeline(scaler, svr)
#
#  print "training"
#  #clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads)
#  #print clf
#  clf.fit(X, y)
#  log_write('clf:' + str(clf))
#  #log_time(t0, 'preprocess data')
#  #print "Best parameters found:"
#  #print clf.best_params_
#
#  score_train = clf.score(X, y)
#  #score_test = clf.score(X_test, y_test)
#  #svr_params = clf.get_params()
#
#  print 'score_train: ' + str(score_train)
#  #print 'score_test: ' + str(score_test)
#  #print 'svr_params: ' + str(svr_params)
#
#  #score = cross_val_score()
#  #print 'training neg_mean_sq_error:' + str(cross_val_score(clf, X, y, scoring='neg_mean_squared_error'))
#
#  #result = { 'clf': clf, 'params':str(clf.best_params_), 'score_train':score_train }
#  result = { 'clf': clf, 'score_train':score_train }
#  return result
 
#------------------------------------------------------------------------------
"""
Training step
"""
def svr_train(args, X, y):
  testing = args.get('testing')
  alg = args.get('algorithm')
  clf = args.get('clf')
  print clf
#  #print params
#
#  scaler = preprocessing.StandardScaler()
#
#  scores = ['precision', 'recall' ]
#  score = scores[0]
#  #svr = LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
#  #   intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
#  #   random_state=0, tol=0.0001, verbose=0)
#  #svr = LinearSVR(random_state=0)
#
#  C=params.get(str(alg+'__C'))
#  eps=params.get(str(alg+'__epsilon'))
#  itr=params.get(str(alg+'__max_iter'))
#  tol=params.get(str(alg+'__tol'))
#
#  svr = LinearSVR(C=C, epsilon=eps, max_iter=itr, tol=tol)
#  
#  #C = [0] 
#  #eps = [1e-1]
#  #max_iter = [1e3]
#  #tol = [1e-5]
#
#  #hyperparameters = { 'linearsvr__C': C, 'linearsvr__epsilon' : eps, 'linearsvr__max_iter' : max_iter, 'linearsvr__tol' : tol }
#  
#  clf = make_pipeline(scaler, svr)

  print "training"
  #clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads)
  #print clf
  clf.fit(X, y)
  log_write('clf:' + str(clf))
  #log_time(t0, 'preprocess data')
  #print "Best parameters found:"
  #print clf.best_params_

  score_train = clf.score(X, y)
  #score_test = clf.score(X_test, y_test)
  #svr_params = clf.get_params()

  print 'score_train: ' + str(score_train)
  #print 'score_test: ' + str(score_test)
  #print 'svr_params: ' + str(svr_params)

  #score = cross_val_score()
  #print 'training neg_mean_sq_error:' + str(cross_val_score(clf, X, y, scoring='neg_mean_squared_error'))

  #result = { 'clf': clf, 'params':str(clf.best_params_), 'score_train':score_train }
  result = { 'clf': clf, 'score_train':score_train }
  return result
 

#------------------------------------------------------------------------------
"""
Testing step
"""
def svr_test(args, svr, X, y):
  clf = args.get('clf')['clf']
  sim = args.get('simulator')
  alg = args.get('algorithm')

  score_test = clf.score(X, y)
  #svr_params = clf.get_params()

  print 'score_test: ' + str(score_test)
  #print 'svr_params: ' + str(svr_params)

  #score = cross_val_score()
  
  ## fit the test set to the regression model
  print 'fitting test data'
  y_pred = clf.predict(X)
  #print y_pred
  r2 = r2_score(y, y_pred)
  mean_sq = mean_squared_error(y, y_pred)

  print 'r2 error (testing): ' + str(r2)
  print 'mean-sq error (testing): ' + str(mean_sq)
  #err = y_pred - y_test
  #mu = np.mean(err)
  #err2 = err * err
  #mu2 = np.mean(err2)
  #print 'error_mu: ' + str(mu)
  #print 'error_mu2: ' + str(mu2)

  #cr = classification_report(y_test, pred)

  #save y-pred
  fname_y_pred = 'svr_' + sim + '_' + alg + '_y-fit_testing.txt'
  np.savetxt(fname_y_pred, y_pred, delimiter=' ', fmt='%1.15e')

  #svr['r2'] = r2
  #svr['mean-sq'] = mean_sq
  return svr
 
#------------------------------------------------------------------------------
"""
"""
def svr_eval_training(args, X, y):
  clf = args.get('clf')['clf']
  sim = args.get('simulator')
  alg = args.get('algorithm')
  print clf

  #score_test = clf.score(X, y)
  #svr_params = clf.get_params()

  #print 'score_test: ' + str(score_test)
  #print 'svr_params: ' + str(svr_params)

  #score = cross_val_score()
  
  ## fit the test set to the regression model
  print 'fitting training data'
  y_pred = clf.predict(X)
  #print y_pred
  r2 = r2_score(y, y_pred)
  mean_sq = mean_squared_error(y, y_pred)

  print 'r2 error (training): ' + str(r2)
  print 'mean-sq error (training): ' + str(mean_sq)
  #err = y_pred - y_test
  #mu = np.mean(err)
  #err2 = err * err
  #mu2 = np.mean(err2)
  #print 'error_mu: ' + str(mu)
  #print 'error_mu2: ' + str(mu2)

  #cr = classification_report(y_test, pred)

  #save y-pred
  fname_y_pred = 'svr_' + sim + '_' + alg + '_y-fit_training.txt'
  np.savetxt(fname_y_pred, y_pred, delimiter=' ', fmt='%1.15e')

  #svr['r2'] = r2
  #svr['mean-sq'] = mean_sq
  #return svr
 
#------------------------------------------------------------------------------
"""
"""
def svr_fit_session(args, svr, session):
  clf = args.get('clf')['clf']
  alg = args.get('algorithm')
  log_write('fitting session-' + session)
  
  data_file = 'trial_' + session + '_testset.txt'
  X = load_data_file( data_file )
  
  y = clf.predict(X)
  #result = svr_fit(args, clf, X)
  
  log_write('samples fit:' + str(len(X)))
  
  fname_y_fit = 'svr_' + sim + '_' + alg + '_y-fit_session-' + session + '.txt'
  np.savetxt(fname_y_fit, y, delimiter=' ', fmt='%1.15e')
  

#------------------------------------------------------------------------------
"""
"""
def svr_fit(args, clf, X):
  ## fit the real data to the regression model
  print 'fitting real data'
  y = clf.predict(X)
  return y

#------------------------------------------------------------------------------
"""
Loads simulation data from disk.  Generates training and test sets from data.
Saves training and test sets to disk
"""
def svr_stage1(args):
  testing = args.get('testing') 
  split_sz = args.get('split-size')
  sim = args.get('simulator')

  X, y = load_simulation_set( args )
  
  log_write('loaded ' + str(len(y)) + ' simulation samples')
  
  #print 'shape X:' + str(X.shape)
  
  if testing:
    ## decimate the data
    print 'decimating'
    X, y = decimate_training_set(X, y, decimation_step)
  
  #print 'shape X:' + str(X.shape)
  
  split_sz = args.get('split-size')
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=split_sz,
                                                      shuffle=True, 
                                                      stratify=y)
  fname_X_train = 'svr_' + sim + '_X-train.txt'
  fname_y_train = 'svr_' + sim + '_y-train.txt'
  fname_X_test = 'svr_' + sim + '_X-test.txt'
  fname_y_test = 'svr_' + sim + '_y-test.txt'

  np.savetxt(fname_X_train, X_train, delimiter=' ', fmt='%1.8e')
  np.savetxt(fname_y_train, y_train, delimiter=' ', fmt='%1.8e')
  np.savetxt(fname_X_test, X_test, delimiter=' ', fmt='%1.8e')
  np.savetxt(fname_y_test, y_test, delimiter=' ', fmt='%1.8e')
  
#------------------------------------------------------------------------------
"""
"""
def svr_stage2(args):
  fname_X_train = 'svr_' + sim + '_X-train.txt'
  fname_y_train = 'svr_' + sim + '_y-train.txt'
  fname_X_test = 'svr_' + sim + '_X-test.txt'
  fname_y_test = 'svr_' + sim + '_y-test.txt'

  X_train = np.loadtxt(fname_X_train) 
  y_train = np.loadtxt(fname_y_train) 
  X_test = np.loadtxt(fname_X_test) 
  y_test = np.loadtxt(fname_y_test) 

  svr = svr_cv(args, X_train, y_train)
  #fname = 'params_' + args['algorithm'] + '.json' 
  #save_dictionary(fname, svr['params'])

  fname = 'obj_' + args['algorithm'] 
  save_obj(fname, svr)

#------------------------------------------------------------------------------
"""
"""
def svr_stage3(args):
  fname_X_train = 'svr_' + sim + '_X-train.txt'
  fname_y_train = 'svr_' + sim + '_y-train.txt'
  fname_X_test = 'svr_' + sim + '_X-test.txt'
  fname_y_test = 'svr_' + sim + '_y-test.txt'

  X_train = np.loadtxt(fname_X_train) 
  y_train = np.loadtxt(fname_y_train) 
  X_test = np.loadtxt(fname_X_test) 
  y_test = np.loadtxt(fname_y_test) 

  #fname = 'params_' + args['algorithm'] + '.json' 
  #args['params'] = load_dictionary(fname)
  #args['params'] = load_dictionary(fname)
  fname = 'obj_' + args['algorithm'] 
  args['clf'] = load_obj(fname)
  #print args['params']

  #svr = svr_train(args, X_train, y_train)
  svr = svr_eval_training(args, X_train, y_train)
  svr = svr_test(args, svr, X_test, y_test)
  #sessions = ['01','02','03','04','05','06','07','08','09','10']
  #sessions = ['01','02']
  sessions = ['01']
  for session in sessions:
    svr_fit_session(args, svr, session)


#------------------------------------------------------------------------------
## main
#trial = '01'
itr = 1e4
decimation_step = 10
cross_validations = 5
split_size = 0.1
threads = 4
testing=False
#testing=True
offsets = [-0.013248,-0.009936,-0.006624,-0.003312, 0.0, 0.003312, 0.006624, 0.009936, 0.013248]

args = {'iterations' : itr, 'decimation-step' : decimation_step, 'cross-validations' : cross_validations, 'threads' : threads, 'split-size': split_size, 'testing':testing, 'offsets':offsets }

stage = int(sys.argv[1])
args['stage'] = stage
sim = sys.argv[2]
args['simulator'] = sim
#if stage == 2:
#  alg = sys.argv[3]    ## only needed if stage 2
#  args['algorithm'] = alg

alg = 'linearsvr'
args['algorithm'] = alg
 
if testing:
  print "WARNING: in test mode.  Optimized for speed instead of accuracy"

#print alg
#print trial
#exit(0)

#if(!(alg == 'rbf' or alg == 'linear' or alg == 'poly')):
#  print("parameter 2 must be rbf, linear, or poly")
#  sys.exit(1);

log = log_start()


log_write('algorithm: ' + str(args.get('algorithm')))
log_write('simulator: ' + str(args.get('simulator')))
log_write('decimation-step: ' + str(args.get('decimation-step')))
#log_write('iterations: ' + str(args.get('iterations')))

if stage == 1:
  # generates training and test sets
  svr_stage1(args)
elif stage == 2:
  # train and fit
  svr_stage2(args)
elif stage == 3:
  # train and fit
  svr_stage3(args)

log_stop(log)
print 'finished'

