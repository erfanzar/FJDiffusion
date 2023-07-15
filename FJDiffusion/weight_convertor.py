def convert_vae_weights(vae_params_flatten, debug: bool = False):
    new_params = {}
    for prm, v in vae_params_flatten.items():
        org_prm = prm
        if prm[0] == 'decoder':

            if prm[1] == 'mid_block':

                if prm[2].startswith('resnets_'):
                    if prm[3] == 'conv1':
                        prm = prm[:3] + ('c1',) + prm[4:]
                    if prm[3] == 'conv2':
                        prm = prm[:3] + ('c2',) + prm[4:]
                    if prm[3] == 'conv_shortcut':
                        prm = prm[:3] + ('cs',) + prm[4:]
                prm = ('decoder', 'bottle_neck') + prm[2:]

            if prm[1].startswith('up_blocks_'):
                num = prm[1][len('up_blocks_'):]
                if prm[2].startswith('resnets_'):
                    if prm[3] == 'conv1':
                        prm = prm[:3] + ('c1',) + prm[4:]
                    if prm[3] == 'conv2':
                        prm = prm[:3] + ('c2',) + prm[4:]
                    if prm[3] == 'conv_shortcut':
                        prm = prm[:3] + ('cs',) + prm[4:]
                prm = ('decoder', f'decoders_{num}') + prm[2:]
            if prm[1] == 'conv_norm_out':
                prm = prm[0:1] + ('out_norm',) + prm[2:]
        if prm[0] == 'encoder':

            if prm[1] == 'mid_block':
                if prm[2].startswith('resnets_'):
                    if prm[3] == 'conv1':
                        prm = prm[:3] + ('c1',) + prm[4:]
                    if prm[3] == 'conv2':
                        prm = prm[:3] + ('c2',) + prm[4:]
                    if prm[3] == 'conv_shortcut':
                        prm = prm[:3] + ('cs',) + prm[4:]
                prm = ('encoder', 'bottle_neck') + prm[2:]

            if prm[1].startswith('down_blocks_'):
                num = prm[1][len('down_blocks_'):]
                if prm[2].startswith('resnets_'):
                    if prm[3] == 'conv1':
                        prm = prm[:3] + ('c1',) + prm[4:]
                    if prm[3] == 'conv2':
                        prm = prm[:3] + ('c2',) + prm[4:]
                    if prm[3] == 'conv_shortcut':
                        prm = prm[:3] + ('cs',) + prm[4:]
                prm = ('encoder', f'encoders_{num}') + prm[2:]
            if prm[1] == 'conv_norm_out':
                prm = prm[0:1] + ('norm_out',) + prm[2:]
        new_params[prm] = v
        if debug:
            print(f'{org_prm} -> {prm}')
    return new_params
