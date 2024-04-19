import re

def train(args):

    new_args = {}

    def extract_numbers(s):
        # Use regular expressions to find all substrings that are numbers
        numbers = re.findall(r'\d+', s)  # '\d+' matches sequences of digits
        # Convert these number strings to integers
        return [int(num) for num in numbers]

    new_args['USE_WANDB'] = args.USE_WANDB
    new_args['wandb_entity'] = args.wandb_entity
    new_args['wandb_project'] = args.wandb_project
    new_args['USE_WANDB'] = args.USE_WANDB
    new_args['outputDir'] = args.outputDir
    new_args['datasetPath'] = args.datasetPath
    new_args['seqLen'] = args.seqLen
    new_args['maxTimeSeriesLen'] = args.maxTimeSeriesLen
    new_args['batchSize'] = args.batchSize
    new_args['lrStart'] = args.lrStart #0.02
    new_args['lrEnd'] = args.lrStart #args.lrEnd #0.02
    new_args['nUnits'] = args.nUnits
    new_args['nBatch'] = args.nBatch #3000
    new_args['nLayers'] = args.nLayers
    new_args['seed'] = args.seed # 0 
    new_args['nClasses'] = args.nClasses
    new_args['nInputFeatures'] = args.nInputFeatures
    new_args['dropout'] = args.dropout
    new_args['whiteNoiseSD'] = args.whiteNoiseSD
    new_args['constantOffsetSD'] = args.constantOffsetSD
    new_args['gaussianSmoothWidth'] = args.gaussianSmoothWidth
    new_args['strideLen'] = extract_numbers(args.strideLen)[0] # 4
    new_args['kernelLen'] = extract_numbers(args.strideLen)[1] #args.kernelLen # 32
    new_args['bidirectional'] = args.bidirectional
    new_args['l2_decay'] = args.l2_decay
    new_args["d_model"] = args.d_model # 256
    new_args["d_state"] = args.d_state
    new_args["d_conv"] = args.d_conv
    new_args["expand_factor"] = args.expand_factor # 4
    new_args['adamBeta2'] =args.adamBeta2 # could try 0.95
    new_args['nWarmup'] = args.nWarmup
    new_args['cosineAnneal'] = args.cosineAnneal
    new_args['lrMin'] = args.lrMin # min for cosine annealing

    from neural_decoder.neural_decoder_trainer_mamba import trainModel

    trainModel(new_args)
