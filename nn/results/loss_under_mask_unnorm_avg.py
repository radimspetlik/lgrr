import numpy as np


def collect_data(json_filepaths):
    data = {}
    for json_filepath in json_filepaths:
        with open(json_filepath, 'r') as f:
            single_file_data = json.load(f)

        data[os.path.splitext(os.path.basename(json_filepath))[0]] = np.array(single_file_data, dtype=np.float)

    return data


def print_table(table_data, argmins, epochs, maps, vals, optimum_finding_function=np.nanmin):
    # feed_mask, ic, reflection_size

    map_0, map_1 = maps
    vals_0, vals_1 = vals

    header = '|| {:10s} '.format('LR \ loss terms')
    map_0_keys_for_header = loss_terms#['{:.2E}'.format(num) for num in vals_0]
    for map_0_key in map_0_keys_for_header:
        header += '||{:s} '.format(map_0_key)
    header += '||'

    print(header)

    value_prefix = r'$'
    value_postfix = r'$'
    row_starter = ''
    separator = '&'
    row_finisher = r'\\'
    maximum_starter = r'\mathbf{' #*
    maximum_finisher = r'}' #*

    head_names = {'{:.0E}'.format(num): '{:.0E}'.format(num) for idx, num in enumerate(vals_1)}
    # head_names = {'l1ld': 'L2', 'l2sq': 'L2 squared'}
    for map_1_key in map_1.keys():
        string_to_print = row_starter + '{:s} ^({:.0f} epochs)^'
        data_to_print = [head_names[map_1_key], epochs[0, map_1[map_1_key]]]
        for map_0_key in map_0.keys():
            string_to_print += separator + ' ' + value_prefix + '{:s}{:.2f}{:s}' + value_postfix + ' '
            is_optimum = optimum_checker(optimum_finding_function, table_data, map_0[map_0_key], map_1[map_1_key])
            data_to_print.append(maximum_starter if is_optimum else '')
            data_to_print.append(table_data[map_0[map_0_key], map_1[map_1_key]])
            to_add = ''
            if is_optimum:
                to_add += maximum_finisher
                if argmins is not None:
                    to_add += '^(epoch {:.0f})^ '.format(argmins[map_0[map_0_key], map_1[map_1_key]])
            data_to_print.append(to_add)
        string_to_print += row_finisher

        print(string_to_print.format(*data_to_print))


def optimum_checker(optimum_finding_function, table_data, key_0, key_1):
    is_optimum = table_data[key_0, key_1] == optimum_finding_function(table_data[:, :])
    return is_optimum


if __name__ == '__main__':
    import json
    import os
    import glob
    import numpy as np
    import re
    from matplotlib import pyplot as plt

    subset = 'val'

    base_path = os.path.join('c:', os.sep, 'Users', 'jarmi', 'data', 'reflections', 'experiments', 'data', subset,
                             'loss_under_mask_unnorm_avg')

    json_filepaths = glob.glob(os.path.join(base_path, '*.json'))

    data = collect_data(json_filepaths)

    # a = 'RR13_ic=4_o=adam_lr=1E-04_bs=34_c=l1=1E+00_fgw=1E+00_bgw=0E+00_lc=100_fm=1_lro=sgd-0.90_lmt=prcs_dsfl2_0.json'
    # with open(os.path.join(base_path, a), 'r') as f:
    #     datas = np.array(json.load(f))
    #     print(datas.shape[0])
    #     print(datas[:, 2].min(axis=0))
    #     exit(0)


    # corr_arch=UpscaleUGSIIRR13_ic=4_lr=1E-03_mom=0E+00_bs=74_c=l1l_wit=0E+00_mt=1E+00_ft=0E+00_mgbs=0E+00_lc=100_fm=1_bof-add_0.4-0.4_lmt=prcs_0

    vals_0 = np.arange(25., 50., 5.) / 100.
    vals_1 = 10 ** np.arange(-3., -2.)

    # map_0 = {'{:.2E}'.format(num): idx for idx, num in enumerate(vals_0)}
    loss_terms = ['l1', 'xy', 'x2', 'x3', 'x4', 'x5', 'x6']
    map_0 = {'{}'.format(lt): idx for idx, lt in enumerate(loss_terms)}
    map_1 = {'{:.0E}'.format(num): idx for idx, num in enumerate(vals_1)}
    # map_1 = {'l1ld': 0}

    argmins = np.ones((len(map_0), len(map_1))) * np.nan
    epochs = np.ones((len(map_0), len(map_1))) * np.nan
    mins = np.ones((len(map_0), len(map_1))) * np.nan
    meds = np.ones((len(map_0), len(map_1))) * np.nan
#r=1.50E-01_lrlr=1E-02
    p = re.compile('.*lr=(.E[+-][0-9]{2}).*(..)=.E.0._bof-satone$')

    basic = re.compile('.*wd=1E-03.*_bof-satone')

    for key, value in data.items():
        if basic.match(key) is None:
            continue

        z = p.match(key)
        key_1, key_0 = z.groups()
        argmins[map_0[key_0], map_1[key_1]] = value[np.argmin(value[:, -1], axis=0), 1]
        epochs[map_0[key_0], map_1[key_1]] = np.max(value[:, 1])
        mins[map_0[key_0], map_1[key_1]] = value[:, -1].min(axis=0)
        meds[map_0[key_0], map_1[key_1]] = np.median(value[:, -1], axis=0)

    # mins = (mins * 100) / 255
    # meds = (meds * 100) / 255

    print('Minimum avg(RGB) error on the foreground [0..255]')
    print_table(mins, argmins, epochs, (map_0, map_1), (vals_0, vals_1))

    print('')
    print('Median AVG(RGB) Error On The Foreground [0..255]')
    print_table(meds, None, epochs, (map_0, map_1), (vals_0, vals_1))
