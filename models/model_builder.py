import os
from models.var import VAR
from models.dnri.models import decoders, nri, encoders


def build_model(params):
    if params['model_type'].lower() == 'var':
        model = VAR(params)
        print("VAR: ", model)
    else:
        num_vars = params['num_vars']
        graph_type = params['graph_type']

        # Build Encoder
        encoder_name = params['encoder_type']
        encoder = getattr(encoders, encoder_name)
        encoder = encoder(params)
        print("ENCODER: ", encoder)

        # Build Decoder
        decoder_name = params['decoder_type']
        decoder = getattr(decoders, decoder_name)
        decoder = decoder(params)
        print("DECODER: ", decoder)

        model = nri.StaticNRI(num_vars, encoder, decoder, params)

    if params['load_best_model']:
        print("LOADING BEST MODEL")
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM SPECIFIED PATH")
        model.load(params['load_model'])
    if params['gpu']:
        model.cuda()
    return model

