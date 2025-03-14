import json 
from matplotlib import pyplot as plt 

with open("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/CIFAR_MADGAN_DATACREATION//labels.json") as f:
    d = json.load(f)

labels = d.values()

plt.hist(labels)

plt.savefig("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/VANILLA_GAN_DATACREATION/2025-03-11___CIFAR_GENERATIVE_VanillaGAN_Experiment_dc_paper_like_epch_15/hist.png")


# for t in ["big", "small", "base"]: 
#     
#     with open(f"/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/CIFAR_MADGAN_DATACREATION_PROTOTYPES/2025-03-07_MADGAN_CIFAR_PROTOTYPE_{t}/labels.json") as f: 
#         d = json.load(f)
# 
#     l = d.values()
# 
#     plt.hist(l)
# 
#     plt.savefig(f"./hist_{t}.png")
# 
#     plt.close()