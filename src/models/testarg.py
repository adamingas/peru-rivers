import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--config', metavar='config-file', type=str, nargs=1,default = "default-default_config.py",
                    help='Name of custom config file',required =False)
#parser.add_argument('--sum', dest='accumulate', action='store_const',
#                    const=sum, default=max,
#                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.config)
