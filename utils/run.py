import os 
import configparser

config = configparser.ConfigParser()
config.read('cfg/settings.cfg')
run_name_prefix = config['log']['run_name_prefix']
runs_dir = config['log']['runs_dir']

# use for auto increase run number in runs folder
def increment_dir(dir = runs_dir, run_name = ""):
    all_file = [i for i in os.listdir(dir) if not i == '.DS_Store']
    if len(all_file) == 0:
        num = 0
    else:
        num = max([int(i[:i.find('_') if '_' in i else None].split(run_name_prefix)[1]) for i in all_file]) + 1
    return dir + '/' + run_name_prefix + str(num) + ( '_' + run_name if run_name else "")