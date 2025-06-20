if __name__ == '__main__':
    import os
    import glob
    import urllib.parse
    import urllib.request

    subset = 'val'

    basepath = os.path.join('c:', os.sep, 'Users', 'jarmi', 'data', 'reflections', 'experiments')
    tensorboard_path = os.path.join(basepath, 'tensorboard_real_imp')

    tensorboard_server_address = 'http://localhost:6069/data/plugin/scalars/scalars?tag={}%2Floss_under_mask_unnorm_avg&run={}&experiment='
    target_json_basepath = os.path.join(basepath, 'data', subset, 'loss_under_mask_unnorm_avg')
    if not os.path.isdir(target_json_basepath):
        os.makedirs(target_json_basepath)

    dirs = glob.glob(os.path.join(tensorboard_path, '*'))
    for dir in dirs:
        basename = os.path.basename(dir)

        address = tensorboard_server_address.format(subset, urllib.parse.quote(basename))

        try:
            contents = urllib.request.urlopen(address).read()
        except urllib.error.HTTPError as e:
            print('Could not get the json for file {}.'.format(basename))
            continue

        target_json_path = os.path.join(target_json_basepath, '{}.json'.format(basename))
        target_json_path = target_json_path.replace('corr_a=UpscaleUGSIIRR', '').replace('_lc=100_fm=0_bof-nosat_0.0-1.0_lmt=prcs_grt=rea_L2', '')
        if os.path.isfile(target_json_path):
            print('The file {} already exists!'.format(basename))
            continue

        with open(target_json_path, 'w') as f:
            f.write(contents.decode('utf8'))