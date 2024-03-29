
modelName = 'MambaInit'

args = {}
args['outputDir'] = '/scratch/users/dzoltow/SpeechBCI/logs/' + modelName
args['datasetPath'] = '/scratch/users/dzoltow/SpeechBCI/competitionData/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.01 #0.02
args['lrEnd'] = 0.01 #0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 13 # 0 
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args["d_model"] = 1024
args["d_state"] = 16
args["d_conv"] = 2
args["expand_factor"] = 1

from neural_decoder.neural_decoder_trainer_mamba import trainModel

trainModel(args)
