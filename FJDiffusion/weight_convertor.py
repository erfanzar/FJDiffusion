def convert_vae_weights_diff_to_fj(vae_params_flatten, debug: bool = False):
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


def convert_unet_weights_diff_to_fj(unet_params_flatten, debug: bool = False):
    new_params = {}
    for prm, v in unet_params_flatten.items():
        org_prm = prm

        if len(prm) > 2:
            if prm[2].startswith('transformer_blocks_'):
                num = prm[2][len('transformer_blocks_'):]
                prm = prm[:2] + ('blocks', f'{num}') + prm[3:]
        if len(prm) > 5:
            if prm[5] == 'to_k':
                prm = prm[:5] + ('k',) + prm[6:]
            if prm[5] == 'to_v':
                prm = prm[:5] + ('v',) + prm[6:]
            if prm[5] == 'to_q':
                prm = prm[:5] + ('q',) + prm[6:]
            if prm[5] == 'to_out_0':
                prm = prm[:5] + ('out',) + prm[6:]
        if len(prm) > 2:
            if prm[1].startswith('resnets_'):
                if prm[2] == 'conv1':
                    prm = prm[:2] + ('c1',) + prm[3:]
                if prm[2] == 'conv2':
                    prm = prm[:2] + ('c2',) + prm[3:]
                if prm[2] == 'conv_shortcut':
                    prm = prm[:2] + ('cs',) + prm[3:]
                if prm[2] == 'time_emb_proj':
                    prm = prm[:2] + ('time_emb',) + prm[3:]

        # if prm[0] == 'bottle_neck':
        #     if prm[1].startswith('resnets_'):
        #         if prm[2] == 'conv1':
        #             prm = prm[:2] + ('c1',) + prm[3:]

        if prm[0].startswith('time_embedding'):
            prm = ('time_emb',) + prm[1:]
            if prm[1] == 'linear_1':
                prm = prm[0:1] + ('l1',) + prm[2:]
            if prm[1] == 'linear_2':
                prm = prm[0:1] + ('l2',) + prm[2:]
        if debug:
            print(f'{org_prm} -> {prm}')
        new_params[prm] = v
    return new_params
