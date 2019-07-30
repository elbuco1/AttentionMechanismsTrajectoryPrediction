import os
import sys

args = sys.argv

dir_name = args[1]

os.system("mkdir ../{}".format(dir_name))
os.system("cp -r {}/src ../{}/src".format(".",dir_name))



os.system("mkdir ../{}/data".format(dir_name))
os.system("mkdir ../{}/data/raw".format(dir_name))
os.system("mkdir ../{}/data/processed".format(dir_name))

os.system("mkdir ../{}/models".format(dir_name))
os.system("mkdir ../{}/models/training".format(dir_name))

os.system("mkdir ../{}/reports".format(dir_name))
os.system("mkdir ../{}/reports/gradients".format(dir_name))
os.system("mkdir ../{}/reports/losses".format(dir_name))

os.system("cp -r {}/data/raw/images ../{}/data/raw/".format(".",dir_name))
os.system("cp ../training.sh ../{}/training.sh".format(dir_name))



print("don't forget to modify training.sh and the parameters of the training")

