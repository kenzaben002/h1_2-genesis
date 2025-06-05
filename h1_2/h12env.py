import torch
import math
import genesis as gs
from genesis.utils.geom import *


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class H1_2Env:
    #initialisation des donner 
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  
        self.dt = 0.02# control frequency on real robot 
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.domain_rand_cfg = domain_rand_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        #ajout de links des pieds 
        #self.feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        #self.feet_num = len(self.feet_names)
        #self.simulation_dt=env_cfg["simulation_dt"]
        #self.control_dt=env_cfg["control_dt"]
        #self.control_decimation=int(self.control_dt/self.simulation_dt)
    
        # creation dù scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=True,
            ),
            show_viewer=show_viewer,
            renderer=gs.renderers.Rasterizer(),
        )

        # add plain
        #terrain1 = gs.morphs.Terrain(
            #pos=(-25.0, -25.0, 0.0),
            #randomize=False, 
            #n_subterrains=(2, 2),
            #subterrain_size=(25.0, 25.0),
            #subterrain_types="pyramid_stairs_terrain")
        #self.scene.add_entity(terrain1)
        
        self.plane=self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file='/home/kbenhammaa/h1_2-genesis/h1_2/h1_2_12dof.urdf',  
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # building the scene 
        print("building the scene ")
        self.scene.build(n_envs=num_envs)
        #cam.start_recording()
        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        

        # PD control parameters
        print("Setting PD control parameters...")
        # Obtention des gains spécifiques à chaque joint
        joint_names = self.env_cfg["joint_names"]
        joint_kps = [self.env_cfg["joint_kps"][name] for name in joint_names]
        joint_kds = [self.env_cfg["joint_kds"][name] for name in joint_names]

        # Application des gains à chaque joint
        self.robot.set_dofs_kp(joint_kps, self.motors_dof_idx)
        self.robot.set_dofs_kv(joint_kds, self.motors_dof_idx)

        #self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        #self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)
        # les memes gains appliquer a tt les joints 

        # prepare reward functions and multiply reward scales by dt
        print("Preparing reward functions...")
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        print("Initializing buffers...")
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        
        ###
        self.contact_forces = self.robot.get_links_net_contact_force()
        self.left_foot_link = self.robot.get_link(name='left_ankle_roll_link')
        self.right_foot_link = self.robot.get_link(
            name='right_ankle_roll_link')
        self.left_foot_id_local = self.left_foot_link.idx_local
        self.right_foot_id_local = self.right_foot_link.idx_local
        self.feet_indices = [self.left_foot_id_local,
                             self.right_foot_id_local]
        self.feet_num = len(self.feet_indices)
        self.links_vel = self.robot.get_links_vel()
        self.feet_vel = self.links_vel[:, self.feet_indices, :]
        self.links_pos = self.robot.get_links_pos()
        self.feet_pos = self.links_pos[:, self.feet_indices, :]
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        self.sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        self.cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.pelvis_link = self.robot.get_link(name='pelvis')
        self.pelvis_mass = self.pelvis_link.get_mass()
        self.pelvis_id_local = self.pelvis_link.idx_local
        self.pelvis_pos = self.links_pos[:, self.pelvis_id_local, :]
        self.original_links_mass = []
        self.counter = 0

        termination_contact_names = self.env_cfg["terminate_after_contacts_on"]
        self.termination_contact_indices = []
        for name in termination_contact_names:
            link = self.robot.get_link(name)
            link_id_local = link.idx_local
            self.termination_contact_indices.append(link_id_local)
    #####################################"retour des cmd
        self.eval_logger = None
        self.eval_data = []
        self.log_columns = [
            'step',
            'cmd_lin_vel_x', 'cmd_lin_vel_y', 'cmd_ang_vel_z',
            'base_pos_x', 'base_pos_y', 'base_pos_z',
            'base_ori_w', 'base_ori_x', 'base_ori_y', 'base_ori_z',
            'base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z',
            'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z'
        ]
        
        # Ajoutez les noms de joints dynamiquement
        joint_names = self.env_cfg["joint_names"]
        for name in joint_names:
            self.log_columns.extend([
                f'{name}_pos',
                f'{name}_vel',
                f'{name}_eff'
            ])
    
    def start_evaluation_logging(self, log_path="eval_log.csv"):
        """À appeler avant de commencer l'évaluation"""
        self.eval_logger = open(log_path, "w")
        self.eval_logger.write(",".join(self.log_columns) + "\n")
        
    def stop_evaluation_logging(self):
        """À appeler après l'évaluation"""
        if self.eval_logger:
            self.eval_logger.close()
        self.eval_logger = None
    
    def _log_evaluation_data(self):
        """Enregistre les données de l'étape actuelle"""
        if not self.eval_logger:
            return
            
        # Données de base
        log_data = {
            'step': self.episode_length_buf[0].item(),
            'cmd_lin_vel_x': self.commands[0, 0].item(),
            'cmd_lin_vel_y': self.commands[0, 1].item(),
            'cmd_ang_vel_z': self.commands[0, 2].item(),
            'base_pos_x': self.base_pos[0, 0].item(),
            'base_pos_y': self.base_pos[0, 1].item(),
            'base_pos_z': self.base_pos[0, 2].item(),
            'base_ori_w': self.base_quat[0, 0].item(),
            'base_ori_x': self.base_quat[0, 1].item(),
            'base_ori_y': self.base_quat[0, 2].item(),
            'base_ori_z': self.base_quat[0, 3].item(),
            'base_lin_vel_x': self.base_lin_vel[0, 0].item(),
            'base_lin_vel_y': self.base_lin_vel[0, 1].item(),
            'base_lin_vel_z': self.base_lin_vel[0, 2].item(),
            'base_ang_vel_x': self.base_ang_vel[0, 0].item(),
            'base_ang_vel_y': self.base_ang_vel[0, 1].item(),
            'base_ang_vel_z': self.base_ang_vel[0, 2].item()
        }
        
        # Données des joints
        joint_pos = self.dof_pos[0].cpu().numpy()
        joint_vel = self.dof_vel[0].cpu().numpy()
        joint_eff = self.robot.get_dofs_force()[0].cpu().numpy()
        
        for i, name in enumerate(self.env_cfg["joint_names"]):
            log_data[f'{name}_pos'] = joint_pos[i]
            log_data[f'{name}_vel'] = joint_vel[i]
            log_data[f'{name}_eff'] = joint_eff[i]
        
        # Écriture dans le fichier
        line = ",".join([str(log_data[col]) for col in self.log_columns])
        self.eval_logger.write(line + "\n")
        self.eval_logger.flush()

        ####################################################################""
    def _resample_commands(self, envs_idx):
        #print("resample")
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        #print("step start ")
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])# intervalle des actions
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions# execution des action selon un delai
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos #determine la pos cible et applique la au robotbv 
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        # For unknown reasons, it gets NaN values in self.robot.get_*() sometimes
        if torch.isnan(self.base_pos).any():
            nan_envs = torch.isnan(self.base_pos).any(dim=1).nonzero(as_tuple=False).flatten()
            self.reset_idx(nan_envs)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            #rpy=True,
            #degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
      
        # resample commands / changement des cmd pour certains env
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # condition de reset 
        self.reset_buf = self.episode_length_buf > self.max_episode_length # depaase le nbr de pas max
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]# detection d'instabilitter 
        self.reset_buf |= torch.abs(self.pelvis_pos[:, 2]) < self.env_cfg["termination_if_pelvis_z_less_than"]
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())# reset pour ces env 
        #domain randomisation
        if(self.domain_rand_cfg['randomize_friction']):
            self.randomize_friction()
        
        if(self.domain_rand_cfg['randomize_mass']):
            self.randomize_mass()

        if(self.domain_rand_cfg['push_robots']):
            self.push_robots()


        #modified physiques 
        self.contact_forces = self.robot.get_links_net_contact_force()
        self.left_foot_link = self.robot.get_link(name='left_ankle_roll_link')
        self.right_foot_link = self.robot.get_link(
            name='right_ankle_roll_link')
        self.left_foot_id_local = self.left_foot_link.idx_local
        self.right_foot_id_local = self.right_foot_link.idx_local
        self.feet_indices = [self.left_foot_id_local,
                             self.right_foot_id_local]
        self.feet_num = len(self.feet_indices)
        self.links_vel = self.robot.get_links_vel()
        self.feet_vel = self.links_vel[:, self.feet_indices, :]
        self.links_pos = self.robot.get_links_pos()
        self.feet_pos = self.links_pos[:, self.feet_indices, :]
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        self.sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        self.cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.extras = {
        "observations": {
            "critic": None  # Initialisé avec None, sera mis à jour dans step()
        },
        "episode": {}  # Pour les métriques d'épisode 
        }

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
             #print(f"[{name}] reward per env: {rew}")


        # compute observations // ce qui envoyer au robot 
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"], 
                self.projected_gravity,  
                self.commands * self.commands_scale,  
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  
                self.dof_vel * self.obs_scales["dof_vel"],  
                self.actions,  
                self.sin_phase,
                self.cos_phase,

            ],
            axis=-1,
        )
        self.obs_buf = torch.clip(self.obs_buf, -self.env_cfg["clip_observations"], self.env_cfg["clip_observations"])

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf
        #print(f"Base height: {self.base_pos[:,2].mean().item():.2f}")
        # Debug print in step():
        #print(f"Contact forces: {self.contact_forces[0, self.feet_indices, 2]}")
        #print(f"Velocity: {self.base_lin_vel[:,0].mean().item():.2f}")
        #####################"
        if hasattr(self, 'eval_logger') and self.eval_logger:
            self._log_evaluation_data()
        #############
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


    def randomize_friction(self):
         if(self.counter % int(self.domain_rand_cfg['push_interval_s']/self.dt) == 0):
            friction_range = self.domain_rand_cfg['friction_range']
            self.robot.set_friction_ratio(
                friction_ratio = friction_range[0] +\
                    torch.rand(self.num_envs, self.robot.n_links) *\
                    (friction_range[1] - friction_range[0]),
                link_indices=np.arange(0, self.robot.n_links))
            self.plane.set_friction_ratio(
                friction_ratio = friction_range[0] +\
                    torch.rand(self.num_envs, self.plane.n_links) *\
                    (friction_range[1] - friction_range[0]),
                link_indices=np.arange(0, self.plane.n_links))

    def randomize_mass(self):
        if(self.counter % int(self.domain_rand_cfg['push_interval_s']/self.dt) == 0):
            added_mass_range = self.domain_rand_cfg['added_mass_range']
            added_mass = float(torch.rand(1).item() * (added_mass_range[1] - added_mass_range[0]) + added_mass_range[0])
            new_mass = max(self.pelvis_mass + added_mass, 0.1)
            self.pelvis_link.set_mass(new_mass)

    def push_robots(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.domain_rand_cfg['push_interval_s']/self.dt) == 0]
        if len(push_env_ids) == 0:
            return  # No environments to push in this step
        max_vel_xy = self.domain_rand_cfg['max_push_vel_xy']
        max_vel_rp = self.domain_rand_cfg['max_push_vel_rp']
        new_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        new_base_abg_vel = torch.zeros_like(self.base_ang_vel)
        new_base_lin_vel[push_env_ids] = gs_rand_float(-max_vel_xy, max_vel_xy, (len(push_env_ids), 3), device=self.device)
        new_base_abg_vel[push_env_ids] = gs_rand_float(-max_vel_rp, max_vel_rp, (len(push_env_ids), 3), device=self.device)
        d_vel_xy = new_base_lin_vel - self.base_lin_vel[:, :3]
        d_vel_rp = new_base_abg_vel - self.base_ang_vel[:, :3]
        d_pos = d_vel_xy * self.dt
        d_pos[:, [2]] = 0
        current_pos = self.robot.get_pos()
        new_pos = current_pos[push_env_ids] + d_pos[push_env_ids]
        self.robot.set_pos(new_pos, zero_velocity=False, envs_idx=push_env_ids)
        d_euler = d_vel_rp * self.dt
        current_euler = self.base_euler
        new_euler = current_euler[push_env_ids] + d_euler[push_env_ids]
        new_quat = xyz_to_quat(new_euler)
        self.robot.set_quat(new_quat, zero_velocity=False, envs_idx=push_env_ids)

    
    
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
            #####
        if hasattr(self, 'log_file'):
            self.log_file.close()
        self.log_file = open("joint_states_log.csv", "a") 
        #####""
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_alive(self):
        alive = (self.base_pos[:, 2] > 1.0).float()
        return alive

    def _reward_gait_contact(self):
        #Récompenser le contact du pied avec le sol quand il est censé être en phase d'appui 
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_gait_swing(self):
        #Récompenser le non-contact du pied pendant la phase de swing (quand il doit être en l’air).
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_swing = self.leg_phase[:, i] >= 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_swing)
        return res
    
    def _reward_feet_swing_height(self):
        #Récompenser une hauteur précise des pieds pendant qu’ils sont en l’air
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3],
                             dim=2) > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - self.reward_cfg[
            "feet_height_target"]) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_contact_no_vel(self):
        #Pied ne glisse pas pendant le contact
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3],
                             dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))
        
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,2,6,8]]), dim=1)
     
    def _reward_knee_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[9,3]]), dim=1)

    def _reward_large_steps(self):
    # Only compute distances between front and back legs during swing
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    # Extract foot positions (x-axis forward)
        feet_pos = self.feet_pos  
    # Get swing mask
        swing_mask = self.leg_phase >= 0.55  # shape: [num_envs, feet_num]
    # Only compute stride for legs in swing phase
        for i in range(0, self.feet_num, 2): 
            j = i + 1
            in_swing = swing_mask[:, i] & swing_mask[:, j] 
        # Horizontal (x-y) distance between foot pair
            foot_delta = feet_pos[:, i, :2] - feet_pos[:, j, :2]  
            step_dist = torch.norm(foot_delta, dim=1)  
            res += step_dist * in_swing  
        return res

    def _reward_ang_vel_xy(self):
        #penalise xy axes base vel
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_dof_vel(self):
        #penalize dof acc
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
