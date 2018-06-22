"""
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
"""
"""
def get_outfile_prefix( sim, alg=None):
  s = 'svr_' + sim
  if alg != None:
    s += '_' + alg
  return s

#------------------------------------------------------------------------------
"""
Save a python dictionary to disk in json format
"""
def save_dictionary( froot, dictionary ):
  with open(froot+'.json', 'w') as f:
    f.write(json.dumps(dictionary))
    f.close()

#------------------------------------------------------------------------------
"""
Load a json formatted dictionary from disk as a python dictionary
"""
def load_dictionary( froot ):
  with open(froot+'.json','r') as f:
    d = json.loads(f.read())
    f.close()
  return d

#------------------------------------------------------------------------------
"""
Pickles an object and saves it to disk.  Unfortunately, joblib does not create
a platform independent format, so the object may not be usable across multiple 
machines.
"""
def save_obj( froot, obj ):
  joblib.dump(obj, froot+'.pkl')

#------------------------------------------------------------------------------
"""
Loads a pickled object from disk
"""
def load_obj( froot ):
  return joblib.load(froot+'.pkl')

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
"""
def svr_linearsvr(args):
  cvs = args.get('cross-validations')
  threads = args.get('threads')

  C = [1] 
  eps = [1e-6]
  max_iter = [1e3]
  tol = [1e-5]

  scaler = preprocessing.StandardScaler()
  svr = LinearSVR(random_state=0)
  hyperparameters = { 'linearsvr__C': C, 'linearsvr__epsilon' : eps, 'linearsvr__max_iter' : max_iter, 'linearsvr__tol' : tol }
  pipeline = make_pipeline(scaler, svr)
  clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, refit=True)
  #print clf

  return clf

#------------------------------------------------------------------------------
"""
"""
def svr_rbf(args):
  cvs = args.get('cross-validations')
  threads = args.get('threads')

  C = [1] 
  eps = [1e-6] 
  gamma = [1e-1]

  scaler = preprocessing.StandardScaler()
  svr = SVR(kernel='rbf')
  hyperparameters = { 'svr__C': C, 'svr__epsilon' : eps, 'svr__gamma' : gamma }
  pipeline = make_pipeline(scaler, svr)
  clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, refit=True)
  #print clf

  return clf

#------------------------------------------------------------------------------
"""
Cross-validation step
"""
def svr_cv(args, clf, X, y):
  testing = args.get('testing')
  alg = args.get('algorithm')

  print "cross-validating"
  t0 = timestamp()
  clf.fit(X, y)
  log_time(t0,"fit to GridSearchCV")

  log_write('clf:' + str(clf))
  #log_time(t0, 'preprocess data')
  print "Best parameters found:"
  print clf.best_params_

  score_train = clf.score(X, y)

  print 'score_train: ' + str(score_train)

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

  print "training"
  clf.fit(X, y)
  log_write('clf:' + str(clf))

  result = { 'clf': clf, 'score_train':score_train }
  return result
 
#------------------------------------------------------------------------------
"""
Report error
"""
def eval_error(dataset, r, y, y_pred):
  err = y_pred - y
  abserr = abs(err)
  mu = np.mean(abserr)
  std = np.std(abserr)
  print dataset+' Error:: mean('+str(mu)+'), pct('+str(mu/r*100)+'%)'
  print 'r2 error ('+dataset+'): ' + str(r2_score(y, y_pred))
  print 'mean-sq error ('+dataset+'): ' + str(mean_squared_error(y, y_pred))

#------------------------------------------------------------------------------
"""
Testing step
"""
def svr_test(args, svr, X, y):
  clf = args.get('clf')['clf']
  sim = args.get('simulator')
  alg = args.get('algorithm')
  offsets = args.get('offsets')

  ## fit the test set to the regression model
  print 'fitting test data'
  y_pred = clf.predict(X)

  eval_error('Testing', offsets[5], y, y_pred)

  #cr = classification_report(y_test, pred)

  #save y-pred
  fname = get_outfile_prefix(sim, alg) + '_y-fit_testing.txt'
  np.savetxt(fname, y_pred, delimiter=' ', fmt='%1.15e')


  return svr
 
#------------------------------------------------------------------------------
"""
"""
def svr_eval_training(args, X, y):
  clf = args.get('clf')['clf']
  sim = args.get('simulator')
  alg = args.get('algorithm')
  offsets = args.get('offsets')
  print clf

  ## fit the test set to the regression model
  print 'fitting training data'
  y_pred = clf.predict(X)

  eval_error('Training', offsets[5], y, y_pred)

  #cr = classification_report(y_test, pred)

  #save y-pred
  fname = get_outfile_prefix(sim, alg) + '_y-fit_training.txt'
  np.savetxt(fname, y_pred, delimiter=' ', fmt='%1.15e')
 
#------------------------------------------------------------------------------
"""
"""
def svr_fit_session(args, svr, session):
  clf = args.get('clf')['clf']
  alg = args.get('algorithm')
  sim = args.get('simulator')

  log_write('fitting session-' + session)
  
  X = load_data_file( 'trial_' + session + '_testset.txt' )
  
  y = clf.predict(X)
  
  log_write('samples fit:' + str(len(X)))
  
  fname = get_outfile_prefix(sim, alg) + '_y-fit_session-' + session + '.txt'
  np.savetxt(fname, y, delimiter=' ', fmt='%1.15e')
  
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

  np.savetxt(get_outfile_prefix(sim) + '_X-train.txt', X_train, delimiter=' ', fmt='%1.8e')
  np.savetxt(get_outfile_prefix(sim) + '_y-train.txt', y_train, delimiter=' ', fmt='%1.8e')
  np.savetxt(get_outfile_prefix(sim) + '_X-test.txt', X_test, delimiter=' ', fmt='%1.8e')
  np.savetxt(get_outfile_prefix(sim) + '_y-test.txt', y_test, delimiter=' ', fmt='%1.8e')
  
#------------------------------------------------------------------------------
"""
"""
def svr_stage2(args):
  sim = args.get('simulator')
  alg = args.get('algorithm')

  X_train = np.loadtxt(get_outfile_prefix(sim) + '_X-train.txt') 
  y_train = np.loadtxt(get_outfile_prefix(sim) + '_y-train.txt') 

  if alg == 'linearsvr':
    clf = svr_linearsvr(args)
  elif alg == 'rbf':
    clf = svr_rbf(args)
  else:
    exit(1)

  clf = svr_cv(args, clf, X_train, y_train)
  #save_dictionary(get_outfile_prefix(sim) + '_params', svr['params'])

  save_obj(get_outfile_prefix(sim, alg) + '_obj', clf)

#------------------------------------------------------------------------------
"""
"""
def svr_stage3(args):
  sim = args.get('simulator')
  alg = args.get('algorithm')
  sessions = args.get('sessions')
 
  X_train = np.loadtxt(get_outfile_prefix(sim) + '_X-train.txt') 
  y_train = np.loadtxt(get_outfile_prefix(sim) + '_y-train.txt') 
  X_test = np.loadtxt(get_outfile_prefix(sim) + '_X-test.txt') 
  y_test = np.loadtxt(get_outfile_prefix(sim) + '_y-test.txt') 

  #args['params'] = load_dictionary(get_outfile_prefix(sim) + '_params')
  #print args['params']

  args['clf'] = load_obj(get_outfile_prefix(sim, alg) + '_obj')

  #svr = svr_train(args, X_train, y_train)
  svr = svr_eval_training(args, X_train, y_train)
  svr = svr_test(args, svr, X_test, y_test)
  for session in sessions:
    svr_fit_session(args, svr, session)

