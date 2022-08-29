import os 
import configparser

config = configparser.ConfigParser()
config.read('cfg/settings.cfg')
log_dir_prefix = config['log']['log_dir_prefix']
overall_log_dir = config['log']['overall_log_dir']

# use for auto increase run number in runs folder
def increment_dir(dir = overall_log_dir, run_name = ""):
    all_file = [i for i in os.listdir(dir) if not i == '.DS_Store']
    if len(all_file) == 0:
        num = 0
    else:
        num = max([int(i[:i.find('_') if '_' in i else None].split(log_dir_prefix)[1]) for i in all_file]) + 1
    return dir + '/' + log_dir_prefix + str(num) + ( '_' + run_name if run_name else "")