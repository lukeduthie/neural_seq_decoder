
modelName = 'Mamba_Run2'

args = {}
args['outputDir'] = '/scratch/users/mkounga/SpeechBCI/logs/' + modelName
args['datasetPath'] = '/scratch/users/mkounga/SpeechBCI/competitionData/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 1e-2 #0.02
args['lrEnd'] = 1e-2 #0.02
args['nUnits'] = 1024
args['nBatch'] = 20000 #3000
args['nLayers'] = 2
args['seed'] = 15 # 0 
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 0.0
args['strideLen'] = 1 # 4
args['kernelLen'] = 1 # 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args["d_model"] = 1024 # 256
args["d_state"] = 16
args["d_conv"] = 4
args["expand_factor"] = 1 # 4
args['adamBeta2'] = 0.999 # could try 0.95
args['nWarmup'] = 1
args['cosineAnneal'] = True
args['lrMin'] = 1e-6 # min for cosine annealing



from neural_decoder.neural_decoder_trainer_mamba import trainModel

trainModel(args)
