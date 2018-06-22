import wbml
import sys

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
sessions = ['01','02','03','04','05','06','07','08','09','10']
#sessions = ['01','02']
#sessions = ['01']

args = {'iterations' : itr, 'decimation-step' : decimation_step, 'cross-validations' : cross_validations, 'threads' : threads, 'split-size': split_size, 'testing':testing, 'offsets':offsets, 'sessions':sessions }

stage = int(sys.argv[1])
args['stage'] = stage
sim = sys.argv[2]
args['simulator'] = sim
#if stage == 2:
#  alg = sys.argv[3]    ## only needed if stage 2
#  args['algorithm'] = alg

alg = 'rbf'
args['algorithm'] = alg
 
if testing:
  print "WARNING: in test mode.  Optimized for speed instead of accuracy"

#print alg
#print trial
#exit(0)

#if(!(alg == 'rbf' or alg == 'linear' or alg == 'poly')):
#  print("parameter 2 must be rbf, linear, or poly")
#  sys.exit(1);

log = wbml.log_start()


wbml.log_write('algorithm: ' + str(args.get('algorithm')))
wbml.log_write('simulator: ' + str(args.get('simulator')))
#wbml.log_write('iterations: ' + str(args.get('iterations')))
if testing:
  wbml.log_write('decimation-step: ' + str(args.get('decimation-step')))

if stage == 1:
  # generates training and test sets
  wbml.svr_stage1(args)
elif stage == 2:
  # train and fit
  wbml.svr_stage2(args)
elif stage == 3:
  # train and fit
  wbml.svr_stage3(args)

wbml.log_stop(log)
print 'finished'

