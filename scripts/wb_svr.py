"""
args
  rbf
  sim[ode, dart]
"""

import numpy as np
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 
import sys
import time

#------------------------------------------------------------------------------
"""
"""
def sec_to_str(t):
  return "%d:%02d:%02d.%03d" % reduce(lambda ll,b : divmod(ll[0],b) + ll[1:], [(t*1000,),1000,60,60])

#------------------------------------------------------------------------------
"""
"""
def timestamp():
  return time.time()

#------------------------------------------------------------------------------
"""
"""
def log_time(t0, msg):
  t1 = timestamp()
  dt = t1-t0
  s = sec_to_str(dt)
  print 'time to ' + msg + ': ' + s

#------------------------------------------------------------------------------
"""
"""
def log_start():
  start_time = timestamp()
  log = {'start_time' : start_time}
  return log

#------------------------------------------------------------------------------
"""
"""
def log_stop(log):
  start_time = log.get('start_time')
  end_time = timestamp()
  delta_t = end_time - start_time;
  s = sec_to_str(delta_t)
  #print 'execution time(s): ' + str(delta_t)
  print 'execution time: ' + s

#------------------------------------------------------------------------------
"""
"""
def log_write(s):
  #log_file = log.get('log_file')
  #log_file.write(s + '\n') 
  print s

#------------------------------------------------------------------------------
"""
"""
def remove_position(x):
  #print x.shape
  a = np.reshape(x, (-1,15))
  a = a[:,3:15]
  #print a.shape
  return np.reshape(a, (-1,24))
  
#------------------------------------------------------------------------------
"""
"""
def load_data_file( fname ):
  X = np.loadtxt(fname) 
  #X = remove_position(np.loadtxt(fname)) 
  return X

#------------------------------------------------------------------------------
"""
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
"""
def load_training_file( fname, yval ):
  X = np.loadtxt(fname) 
  #X = remove_position(np.loadtxt(fname)) 
  y = np.empty([len(X)])
  y.fill(yval)
  return X, y

#------------------------------------------------------------------------------
"""
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
"""
def decimate_training_set(X, y, step=10):
  idxs = np.arange(0,len(X),step)
  X = np.take(X, idxs, axis=0)
  y = np.take(y, idxs, axis=0)
  return X, y

#------------------------------------------------------------------------------
"""
"""
def decimate_test_set(X, step=10):
  idxs = np.arange(0,len(X),step)
  X = np.take(X, idxs)
  return X

#------------------------------------------------------------------------------
"""
"""
def shuffle_data( Xin, yin ):
  idxs = np.arange(len(Xin))
  np.random.shuffle(idxs)
  X = np.take(Xin, idxs, axis=0)
  y = np.take(yin, idxs, axis=0)
  return X, y, idxs

#------------------------------------------------------------------------------
"""
"""
#def save_y_data(pfx, sfx, ode_y, dart_y):
#  np.savetxt(pfx+'_ode_' + sfx + '.txt', ode_y, delimiter=' ', fmt='%1.15e')
#  np.savetxt(pfx+'_dart_' + sfx + '.txt', dart_y, delimiter=' ', fmt='%1.15e')

#------------------------------------------------------------------------------
"""
"""
def save_y_data(fname, y):
  np.savetxt(fname, y, delimiter=' ', fmt='%1.15e')

#------------------------------------------------------------------------------
"""
"""
def get_y_data_filename(pfx, sfx, sim):
  return pfx+'_' + sim + '_' + sfx + '.txt'

#------------------------------------------------------------------------------
def svr_preprocess(args, X_raw, y_raw):
  alg = args.get('algorithm')
  split_sz = args.get('split-size')
  #cvs = args.get('cross-validations')
  threads = args.get('threads')

  ## preprocessing
  print 'preprocessing'
  t0 = timestamp()
  X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, 
                                                      test_size=split_sz,
                                                      shuffle=True, 
                                                      stratify=y_raw)
   
  #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  #svr_lin = SVR(kernel='linear', C=1e3, epsilon=0.2)
  #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
  #y_rbf = svr_rbf.fit(X_train_scaled, y).predict(X_test_scaled)
  #y_lin = svr_lin.fit(X_train_scaled, y).predict(X_test_scaled)
  #y_poly = svr_poly.fit(X_train_scaled, y).predict(X_test_scaled)
  #svr = SVR()
  #print svr.get_params()
 
  ## NOTE: May need to change scaler 
  scaler = preprocessing.StandardScaler()

  #hyperparameters = [ { 'kernel': ['rbf'], 'gamma': [1e-3,1e-2,1e-1], 'C': [1,1e1,1e2,1e3] } ,{ 'kernel': ['linear'], 'C': [1,1e1,1e2,1e3] } ]

  scores = ['precision', 'recall' ]
  score = scores[0]
  if alg == 'rbf':
    C = [1e-4, 1e-3, 1e-2, 1e-1, 1] 
    eps = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] 
    gamma = [1e-3, 1e-2, 1e-1, 1, 1e1]

    ## two hours
    #C = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]  # 1e-1
    #eps = [1e-3, 1e-4, 1e-5, 1e-6]      # 1e-3, 1e-4, 1e-5
    #gamma = [1e-3, 1e-2, 1e-1, 1]       # 1e-1

    ## best - 1/2 hour
    #C = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3] 
    #eps = [1e-1, 1e-2, 1e-3] 
    #gamma = [1e-1, 1, 1e1]

    ## fast - 10 min
    #C = [1e-3, 1e-2, 1e-1, 1] 
    #eps = [1e-1, 1e-2, 1e-3] 
    ##gamma = ['auto']
    #gamma = [1e-1, 1, 1e1]

    svr = SVR(kernel='rbf')
    hyperparameters = { 'svr__C': C, 'svr__epsilon' : eps, 'svr__gamma' : gamma }
  elif alg == 'linear':
    svr = SVR(kernel='linear', max_iter=1e6)
    hyperparameters = { 'svr__C': [1, 1e-1, 1e-2, 1e-3], 'svr__epsilon' : [1e-1, 1e-2, 1e-3]}
  elif alg == 'poly':
    degree = [1,2,3] 

    svr = SVR(kernel='poly', max_iter=1e6)
    hyperparameters = { 'svr__C': [1, 1e-1, 1e-2, 1e-3], 'svr__epsilon' : [1e-1, 1e-2, 1e-3], 'svr__degree' : [1,2,3]}
  else:
    sys.exit(1)

  #hyperparameters = { 'svr__degree' : [2, 3, 4], 'svr__C': [1e3, 1e5, 1e7], 'svr__epsilon' : [0.1], 'svr__max_iter' : [1e4, 1e5, 1e6], 'svr__tol' : [0.001, 0.0001]}
  
  pipeline = make_pipeline(scaler, svr)

  print "cross-validating"
  #clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, scoring='%s_macro' % score)
  #clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, scoring='%s_macro' % score)
  clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads)
  #clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, refit=True)
  #print clf
  clf.fit(X_train, y_train)
  log_write('clf:' + str(clf))
  log_time(t0, 'preprocess data')
  print "Best parameters found:"
  print clf.best_params_

  score_train = clf.score(X_train, y_train)
  score_test = clf.score(X_test, y_test)
  #svr_params = clf.get_params()

  print 'score_train: ' + str(score_train)
  print 'score_test: ' + str(score_test)
  #print 'svr_params: ' + str(svr_params)

  #score = cross_val_score()
  
  ## fit the test set to the regression model
  print 'fitting test data'
  y_pred = clf.predict(X_test)
  #print y_pred
  r2 = r2_score(y_test, y_pred)
  mean_sq = mean_squared_error(y_test, y_pred)
  #err = y_pred - y_test
  #mu = np.mean(err)
  #err2 = err * err
  #mu2 = np.mean(err2)
  #print 'error_mu: ' + str(mu)
  #print 'error_mu2: ' + str(mu2)

  print cross_val_score(clf, X_train, y_train, scoring='neg_mean_squared_error')

  #cr = classification_report(y_test, pred)

  result = { 'clf': clf, 'r2': r2, 'mean-sq': mean_sq, 'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test': y_test, 'y_pred': y_pred }
  return result
 
#------------------------------------------------------------------------------
"""
"""
def svr_cv(args, X, y):
  testing = args.get('testing')
  alg = args.get('algorithm')
  cvs = args.get('cross-validations')

  #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  #svr_lin = SVR(kernel='linear', C=1e3, epsilon=0.2)
  #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
  #y_rbf = svr_rbf.fit(X_train_scaled, y).predict(X_test_scaled)
  #y_lin = svr_lin.fit(X_train_scaled, y).predict(X_test_scaled)
  #y_poly = svr_poly.fit(X_train_scaled, y).predict(X_test_scaled)
  #svr = SVR()
  #print svr.get_params()
 
  ## NOTE: May need to change scaler 
  scaler = preprocessing.StandardScaler()

  #hyperparameters = [ { 'kernel': ['rbf'], 'gamma': [1e-3,1e-2,1e-1], 'C': [1,1e1,1e2,1e3] } ,{ 'kernel': ['linear'], 'C': [1,1e1,1e2,1e3] } ]

  scores = ['precision', 'recall' ]
  score = scores[0]
  if alg == 'rbf':
    if testing:
      ## fast - 10 min
      C = [1e-3, 1e-2, 1e-1, 1] 
      eps = [1e-1, 1e-2, 1e-3] 
      ##gamma = ['auto']
      gamma = [1e-1, 1, 1e1]
    else:
      C = [1e-1, 1] 
      eps = [1e-6] 
      gamma = [1e-1]

      # training error < 5%
      #C = [1e-1, 1, 1e1] 
      #eps = [1e-5, 1e-4, 1e-3] 
      #gamma = [1e-1, 1, 1e1]

      #C = [1e-4, 1e-3, 1e-2, 1e-1, 1] 
      #eps = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] 
      #gamma = [1e-3, 1e-2, 1e-1, 1, 1e1]

    svr = SVR(kernel='rbf')
    hyperparameters = { 'svr__C': C, 'svr__epsilon' : eps, 'svr__gamma' : gamma }
  elif alg == 'linear':
    svr = SVR(kernel='linear', max_iter=1e6)
    hyperparameters = { 'svr__C': [1, 1e-1, 1e-2, 1e-3], 'svr__epsilon' : [1e-1, 1e-2, 1e-3]}
  elif alg == 'poly':
    degree = [1,2,3] 

    svr = SVR(kernel='poly', max_iter=1e6)
    hyperparameters = { 'svr__C': [1, 1e-1, 1e-2, 1e-3], 'svr__epsilon' : [1e-1, 1e-2, 1e-3], 'svr__degree' : [1,2,3]}
  else:
    sys.exit(1)

  #hyperparameters = { 'svr__degree' : [2, 3, 4], 'svr__C': [1e3, 1e5, 1e7], 'svr__epsilon' : [0.1], 'svr__max_iter' : [1e4, 1e5, 1e6], 'svr__tol' : [0.001, 0.0001]}
  
  pipeline = make_pipeline(scaler, svr)

  print "cross-validating"
  #clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, scoring='%s_macro' % score)
  #clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, scoring='%s_macro' % score)
  clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads)
  #clf = GridSearchCV(pipeline, hyperparameters, cv=cvs, n_jobs=threads, refit=True)
  #print clf
  clf.fit(X, y)
  log_write('clf:' + str(clf))
  #log_time(t0, 'preprocess data')
  print "Best parameters found:"
  print clf.best_params_

  score_train = clf.score(X, y)
  #score_test = clf.score(X_test, y_test)
  #svr_params = clf.get_params()

  print 'score_train: ' + str(score_train)
  #print 'score_test: ' + str(score_test)
  #print 'svr_params: ' + str(svr_params)

  #score = cross_val_score()
  #print 'training neg_mean_sq_error:' + str(cross_val_score(clf, X, y, scoring='neg_mean_squared_error'))

  result = { 'clf': clf, 'params':str(clf.best_params_), 'score_train':score_train }
  return result
 
  
#------------------------------------------------------------------------------
"""
"""
def svr_test(args, svr, X, y):
  clf = svr.get('clf')
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
def svr_eval_training(args, svr, X, y):
  clf = svr.get('clf')
  sim = args.get('simulator')
  alg = args.get('algorithm')

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
  return svr
 
#------------------------------------------------------------------------------
"""
"""
def svr_fit_session(args, svr, session):
  clf = svr.get('clf')
  alg = args.get('algorithm')
  log_write('fitting session-' + session)
  
  data_file = 'trial_' + session + '_testset.txt'
  X = load_data_file( data_file )
  
  y = clf.predict(X)
  #result = svr_fit(args, clf, X)
  #ode_result = svr_fit(log, args, ode_pp.get('clf'), X_real)
  #dart_result = svr_fit(log, args, dart_pp.get('clf'), X_real)
  
  log_write('samples fit:' + str(len(X)))
  
  #ode_err = ode_pp.get('y_pred') - ode_pp.get('y_test')
  #dart_err = dart_pp.get('y_pred') - dart_pp.get('y_test')
  #ode_abserr = abs(ode_err)
  #dart_abserr = abs(dart_err)
  #ode_mu = np.mean(ode_abserr)
  #dart_mu = np.mean(dart_abserr)
  #ode_std = np.std(ode_abserr)
  #dart_std = np.std(dart_abserr)
  #err = pp.get('y_pred') - pp.get('y_test')
  #abserr = abs(err)
  #mu = np.mean(abserr)
  #std = np.std(abserr)
  
  #log_write('ode-mu(training error):' + str(ode_mu))
  #log_write('ode-std(training error):' + str(ode_std))
  #log_write('dart-mu(training error):' + str(dart_mu))
  #log_write('dart-std(training error):' + str(dart_std))
  #log_write('offset:' + str(offsets[5]))
  #log_write(sim + '-mu(training error):' + str(mu))
  #log_write(sim + '-std(training error):' + str(std))
  #log_write('offset:' + str(offsets[5]))

  fname_y_fit = 'svr_' + sim + '_' + alg + '_y-fit_session-' + session + '.txt'
  np.savetxt(fname_y_fit, y, delimiter=' ', fmt='%1.15e')
  
  #fname = get_y_data_filename( alg, 'trial-' + session + '_y-fit', sim)
  #save_y_data(alg, 'trial-' + trial + '_y-fit', ode_result, dart_result)
  #save_y_data(fname, y)

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
  
  #print 'shape ode(X:' + str(ode_X.shape) + ', y:' + str(ode_y.shape) + ')'
  #print 'shape dart(X:' + str(dart_X.shape) + ', y:' + str(dart_y.shape) + ')'
  #print 'shape X:' + str(X.shape)
  
  if testing:
    ## decimate the data
    print 'decimating'
    X, y = decimate_training_set(X, y, decimation_step)
  
  #print 'shape ode(X:' + str(ode_X.shape) + ', y:' + str(ode_y.shape) + ')'
  #print 'shape dart(X:' + str(dart_X.shape) + ', y:' + str(dart_y.shape) + ')'
  #print 'shape X:' + str(X.shape)
  
  #log_write('')
  #t0 = timestamp()

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
  svr = svr_eval_training(args, svr, X_train, y_train)
  svr = svr_test(args, svr, X_test, y_test)
  #sessions = ['01','02','03','04','05','06','07','08','09','10']
  #sessions = ['01','02']
  sessions = ['01']
  for session in sessions:
    svr_fit_session(args, svr, session)

##  #pp = svr_preprocess(args, X, y)
##  #log_write('')
##  #log_time(t0, 'fit ' + sim + ' data')
##  #log_write(sim + '-r2-score:' + str(pp.get('r2')))
##  #log_write(sim + '-mean-sq:' + str(pp.get('mean-sq')))
##  
##  #fname = alg + '_y-pred.txt' 
##  #save_y_data(fname, pp.get('y_pred'))
##  #fname = alg + '_y-test.txt' 
##  #save_y_data(fname, pp.get('y_test'))
##  
##  # fit and plot
##  #trials = ['01','02','03','04','05','06','07','08','09','10']
##  #trials = ['01','02']
##  trials = ['01']
##  for trial in trials:
##    log_write('fitting trial-' + trial)
##  
##    data_file = 'trial_' + trial + '_testset.txt'
##    X_real = load_data_file( data_file )
##  
##    result = svr_fit(args, pp.get('clf'), X_real)
##    #ode_result = svr_fit(log, args, ode_pp.get('clf'), X_real)
##    #dart_result = svr_fit(log, args, dart_pp.get('clf'), X_real)
##  
##    log_write('samples fit:' + str(len(X_real)))
##  
##    #ode_err = ode_pp.get('y_pred') - ode_pp.get('y_test')
##    #dart_err = dart_pp.get('y_pred') - dart_pp.get('y_test')
##    #ode_abserr = abs(ode_err)
##    #dart_abserr = abs(dart_err)
##    #ode_mu = np.mean(ode_abserr)
##    #dart_mu = np.mean(dart_abserr)
##    #ode_std = np.std(ode_abserr)
##    #dart_std = np.std(dart_abserr)
##    err = pp.get('y_pred') - pp.get('y_test')
##    abserr = abs(err)
##    mu = np.mean(abserr)
##    std = np.std(abserr)
##  
##    #log_write('ode-mu(training error):' + str(ode_mu))
##    #log_write('ode-std(training error):' + str(ode_std))
##    #log_write('dart-mu(training error):' + str(dart_mu))
##    #log_write('dart-std(training error):' + str(dart_std))
##    #log_write('offset:' + str(offsets[5]))
##    log_write(sim + '-mu(training error):' + str(mu))
##    log_write(sim + '-std(training error):' + str(std))
##    log_write('offset:' + str(offsets[5]))
##  
##    fname = get_y_data_filename( alg, 'trial-' + trial + '_y-fit', sim)
##    #save_y_data(alg, 'trial-' + trial + '_y-fit', ode_result, dart_result)
##    save_y_data(fname, result)

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
if stage == 2:
  alg = sys.argv[3]    ## only needed if stage 2
  args['algorithm'] = alg
  
if testing:
  print "WARNING: in test mode.  Optimized for speed instead of accuracy"

#print alg
#print trial
#exit(0)

#sim = sys.argv[1]
#alg = sys.argv[2]
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

## NOTE: Replaced by Stage 1
#### load data
##print 'loading data'
##X, y = load_simulation_set( args )
##
##log_write('training on ' + str(len(y)) + ' samples')
##
###NOTE: will probably need to decimate this data further.  Best to do it here
### than in the files themselves.  Basically, subsample each of the trainingsets
##
###NOTE: position is probably only complicating fitting.  Position is probably
### irrelevant because max speed occurs when the WB is rolling freely, but this
### can only happen when the WB is not encumbered by a collision.  In the real
### set, these edge cases have been effectively filtered out.  Having the 
### additional parameters in the sets effectively increases the dimensionality
### of the polynomial without adding relevant information.  Should try to filter
### out positional data from sets.  NOTE: Cannot filter position alone and improve
### fit.  Would need to add more information to the data like second derivative
##
###NOTE: for similar reasons, the rotational velocity should be constant, so that
### column may not be relevant
##
###print 'shape ode(X:' + str(ode_X.shape) + ', y:' + str(ode_y.shape) + ')'
###print 'shape dart(X:' + str(dart_X.shape) + ', y:' + str(dart_y.shape) + ')'
###print 'shape X_real:' + str(X_real.shape)
##
##if testing:
##  ## decimate the data
##  print 'decimating'
##  #ode_X, ode_y = decimate_training_set(ode_X, ode_y, decimation_step)
##  #dart_X, dart_y = decimate_training_set(dart_X, dart_y, decimation_step)
##  X, y = decimate_training_set(X, y, decimation_step)
##
###print 'shape ode(X:' + str(ode_X.shape) + ', y:' + str(ode_y.shape) + ')'
###print 'shape dart(X:' + str(dart_X.shape) + ', y:' + str(dart_y.shape) + ')'
###print 'shape X_real:' + str(X_real.shape)
##
###Note: may need to break shuffle into another preprocessing step
### The data would be saved after shuffle into files which are loaded at the
### beginning of this script.  Reshuffling each time this script is run may 
### change tuning which may complicate pipeline.
##
#### preprocess
###t0 = timestamp()
###ode_pp = svr_preprocess(log, args, ode_X, ode_y)
###log_time(t0, 'fit ode data')
###log_write('ode-r2-score:' + str(ode_pp.get('r2')))
###log_write('ode-mean-sq:' + str(ode_pp.get('mean-sq')))
###
###t0 = timestamp()
###dart_pp = svr_preprocess(log, args, dart_X, dart_y)
###log_time(t0, 'fit dart data')
###log_write('dart-r2-score:' + str(dart_pp.get('r2')))
###log_write('dart-mean-sq:' + str(dart_pp.get('mean-sq')))
##
##log_write('')
##t0 = timestamp()
## NOTE: END Stage 1 Replacement

## NOTE: Replaced by Stage 1
##pp = svr_preprocess(args, X, y)
##log_write('')
##log_time(t0, 'fit ' + sim + ' data')
##log_write(sim + '-r2-score:' + str(pp.get('r2')))
##log_write(sim + '-mean-sq:' + str(pp.get('mean-sq')))
##
##fname = alg + '_y-pred.txt' 
##save_y_data(fname, pp.get('y_pred'))
##fname = alg + '_y-test.txt' 
##save_y_data(fname, pp.get('y_test'))
##
### fit and plot
##trials = ['01','02','03','04','05','06','07','08','09','10']
###trials = ['01','02']
###trials = ['01']
##for trial in trials:
##  log_write('fitting trial-' + trial)
##
##  data_file = 'trial_' + trial + '_testset.txt'
##  X_real = load_data_file( data_file )
##
##  result = svr_fit(args, pp.get('clf'), X_real)
##  #ode_result = svr_fit(log, args, ode_pp.get('clf'), X_real)
##  #dart_result = svr_fit(log, args, dart_pp.get('clf'), X_real)
##
##  log_write('samples fit:' + str(len(X_real)))
##
##  #ode_err = ode_pp.get('y_pred') - ode_pp.get('y_test')
##  #dart_err = dart_pp.get('y_pred') - dart_pp.get('y_test')
##  #ode_abserr = abs(ode_err)
##  #dart_abserr = abs(dart_err)
##  #ode_mu = np.mean(ode_abserr)
##  #dart_mu = np.mean(dart_abserr)
##  #ode_std = np.std(ode_abserr)
##  #dart_std = np.std(dart_abserr)
##  err = pp.get('y_pred') - pp.get('y_test')
##  abserr = abs(err)
##  mu = np.mean(abserr)
##  std = np.std(abserr)
##
##  #log_write('ode-mu(training error):' + str(ode_mu))
##  #log_write('ode-std(training error):' + str(ode_std))
##  #log_write('dart-mu(training error):' + str(dart_mu))
##  #log_write('dart-std(training error):' + str(dart_std))
##  #log_write('offset:' + str(offsets[5]))
##  log_write(sim + '-mu(training error):' + str(mu))
##  log_write(sim + '-std(training error):' + str(std))
##  log_write('offset:' + str(offsets[5]))
##
##  fname = get_y_data_filename( alg, 'trial-' + trial + '_y-fit', sim)
##  #save_y_data(alg, 'trial-' + trial + '_y-fit', ode_result, dart_result)
##  save_y_data(fname, result)
## NOTE: END Stage 1 Replacement

## end
log_stop(log)
print 'finished'

