# h1_2-genesis

Objective of this repo is exploring and testing GENSIS IA to gain a deeper understanding of its capabilities in robotics simulation and reinforcement learning using h1_2 unitree robot 

1. Installation and Setup
   
Tested Platforms:
  -Linux (Ubuntu 24.04.2 LTS): Fully functional
  
  -Windows: Works, but might face some rendering issues (OpenGL) → Linux is strongly recommended
  
Requirements:
  -Install torch______with #pip install torch
      (Make sure to install the version of Python and torch suitable for your system)
      
  -Install Genesis____with #pip install genesis-world
  
  -pip install open3d  # For morph operations
   ###For better understanding, please refer to the official documentation:
        https://genesis-world.readthedocs.io/en/latest/user_guide

Notes:
  -Genesis runs more efficiently with a GPU.
    Check your NVIDIA drivers with the command: nvidia-smi
    
  -If the driver is not found, install it:
    sudo apt install nvidia-driver-<version>
    
  -Verify your CUDA version.
  
  -Verify your GPU architecture:
    It must have Compute Capability (C.C) greater than or equal to 3.7.
      #####Some architectures are no longer supported by PyTorch.
         ####For example, the NVIDIA GeForce GTX 780 has a C.C of 3.5 and is therefore not compatible.

2.My system environment

  **Python** : 3.12.3  
  
  **OS** : Ubuntu 24.04.2 LTS 
  
  **CPU** : Intel(R) Core(TM) i9-10900K CPU @ 3.70 GHz 
  
  **GPU** : NVIDIA GeForce RTX 3060  
  
  **Driver** : 550.120  
  
  **CUDA** : 12.4  
  
  **PyTorch** : 2.7.0+cu118  

#####################################

URDF/MJCF files of  Unitree H1_2 qre from this repo :
https://github.com/unitreerobotics/unitree_rl_gym/tree/main/resources/robots/h1_2

   
