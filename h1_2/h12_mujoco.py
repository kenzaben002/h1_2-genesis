import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from pathlib import Path
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
import pickle
import json
import os

class H1Sim2SimTransfer:
    def __init__(self, model_path, policy_path):
        # Initialisation MuJoCo
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002  # Pas de temps fixe 

        # Configuration des articulations
        self._setup_joints()
        
        # Chargement de la policy
        self.policy_path = policy_path
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        
        
        # Paramètres de contrôle
        self.action_scale = 0.2
        self.default_angles = np.array([
            0.0, -0.16, 0.0, 0.36, -0.2, 0.0,  # Jambe gauche   
            0.0, -0.16, 0.0, 0.36, -0.2, 0.0,  # Jambe droite
           
        ])
        self.simulate_action_latency=True
        self.last_actions=np.zeros(len(self.joint_indices))

        # Initialisation des buffers
        self.obs = np.zeros(47)  # Taille d'observation pour 12 joints
        self.actions = np.zeros(len(self.joint_indices))
        self.debug_log = []
        self.counter=0
        ##
        

    def _setup_joints(self):
        """Configure les articulations contrôlées"""
        self.joint_names = [
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            
        ]
        
        self.joint_indices = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                            for name in self.joint_names]
        self.motor_indices = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                            for name in self.joint_names]
        
           
        # Gains PD Kp et Kd 
        self.kps = np.array([200, 200, 200, 300, 40, 40, 
                            200, 200, 200, 300, 40, 40])
        self.kds = np.array([2.5, 2.5, 2.5, 4, 2.0, 2.0,
                            2.5, 2.5, 2.5, 4, 2.0, 2.0])

    def get_observations(self):


        """Calcule toutes les observations avec vérification des noms de joints"""
        
        # Affiche la correspondance nom/index pour qpos[7:]
        #print("\nJoints dans qpos[7:]:")
        for i in range(7, 7+len(self.joint_names)):
            joint_id = i - 7  # Index dans joint_names
            joint_name = self.joint_names[joint_id]
            joint_value = self.data.qpos[i]
            
    
        # Affiche la correspondance nom/index pour qvel[6:] 
        #print("\nJoints dans qvel[6:]:")
        for i in range(6, 6+len(self.joint_names)):
            joint_id = i - 6  # Index dans joint_names
            joint_name = self.joint_names[joint_id]
            joint_vel = self.data.qvel[i]
            
            """Calcule toutes les observations """
        # 1. Vitesse angulaire (3D)
        omega = self.data.qvel[3:6] * 0.25  # ang_vel_scale=0.25

        # 2. Vecteur gravité projeté (3D)
        quat = self.data.qpos[3:7]
        gravity = self._compute_gravity_vector(quat)

        # 3. Commandes (3D)
        commands = np.array([0.5, 0.0, 0.0]) * np.array([2.0, 2.0, 0.25])  # cmd_scale
        
        # 4. Positions articulaires (12D)
        joint_pos = (self.data.qpos[7:] - self.default_angles) * 1.0  

        # 5. Vitesses articulaires (12D)
        joint_vel = self.data.qvel[6:] * 0.05 

        # 6. Actions précédentes (12D)
        last_actions = np.clip(self.actions,-100,100) # clip_actions=100
        
        # 7. Phase de marche (2D)
        period=0.8
        phase = (self.counter*self.model.opt.timestep) % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        
        # Construction de l'observation finale (47D)
        return np.concatenate([
            omega,        # 3
            gravity,      # 3
            commands,     # 3
            joint_pos,    # 12
            joint_vel,    # 12
            last_actions, # 12
            np.array([sin_phase, cos_phase]) #2
        ])

    def _compute_gravity_vector(self, quaternion):
        """Calcule le vecteur gravité à partir du quaternion d'orientation"""
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def pd_control(self,target_q, q, kp, target_dq, dq, kd):
        error = target_q - q
        error_dq = target_dq - dq
        torque = kp * error + kd *error_dq 
    
        return torque

    

    def run(self, duration=20):
        """Exécute la boucle principale de simulation"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            
            while viewer.is_running() and (time.time() - start_time < duration):
                step_start = time.time()
                
                # 1. Mise à jour des observations
                self.obs = self.get_observations()
                
                # 2. Inférence de la policy
                with torch.no_grad():
                    #obs_tensor = torch.FloatTensor(self.obs).unsqueeze(0)
                    #self.actions = self.policy(obs_tensor).numpy().squeeze()
                    obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    self.actions = self.policy(obs_tensor).cpu().numpy().squeeze()
                  
                
                # 3. Calcul des positions cibles
                exec_actions = self.last_actions if self.simulate_action_latency else self.actions
                target_pos = exec_actions * self.action_scale + self.default_angles
                self.last_actions=self.actions.copy()
                #self.actions=exec_actions
                #### normeliser [-1,1]
                
                # 4. Contrôle PD
                q = self.data.qpos[7:]
                dq = self.data.qvel[6:]
                torque = self.pd_control(target_pos, q, self.kps, np.zeros_like(self.kds), dq, self.kds)  #ou zeros_like(dq)
                
                #valeur du torque
                self.data.ctrl[:] = torque
                
                # 5. Log de données
                if self.counter < 10:
                    debug_data = {
                        'timestep': self.counter,
                        'observations': {
                            'angular_velocity': self.obs[0:3].tolist(),
                            'projected_gravity': self.obs[3:6].tolist(),
                            'commands': self.obs[6:9].tolist(),
                            'joint_positions': self.obs[9:21].tolist(),
                            'joint_velocities': self.obs[21:33].tolist(),
                            'previous_actions': self.obs[33:45].tolist(),
                            'phase': {
                                'sin': self.obs[45].item(),
                                'cos': self.obs[46].item()
                            }
                        },

                        'actions': self.actions.tolist(),
                        'target_joint_positions': target_pos.tolist(),
                        'joint_pos': self.data.qpos[7:].tolist(),
                        'joint_vel': self.data.qvel[6:].tolist(),
                        'base_pos': self.data.qpos[:3].tolist(),
                        'base_quat': self.data.qpos[3:7].tolist(),

                    }
                self.debug_log.append(debug_data)

                # 5. Step de simulation
                mujoco.mj_step(self.model, self.data)
                self.counter += 1

                # 6. Synchronisation du viewer
                viewer.sync()
                
                # 7. Contrôle du temps réel
                elapsed = time.time() - step_start
                time.sleep(max(0, self.model.opt.timestep - elapsed))

            output_path = Path("sim_debug_data_10steps.json")
            with open(output_path, "w") as f:
                json.dump(self.debug_log, f, indent=2)
            print(f"Données sauvegardées dans : {output_path}")

if __name__ == "__main__":
    # Configuration des chemins 
    MODEL_PATH = "/home/kbenhammaa/h1_2-genesis/h1_2/scene.xml"
    POLICY_PATH = "/home/kbenhammaa/h1_2-genesis/h1_2/logs/h1_2-walking-v17/policy.jit"
    
    # Lancement de la simulation
    simulator = H1Sim2SimTransfer(MODEL_PATH, POLICY_PATH)
    
    
    # Exécution
    simulator.run(duration=200)
