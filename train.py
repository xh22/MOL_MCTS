from rdkit.Chem import AllChem as Chem
import numpy as np
import pandas as pd
from molecules import  Molecule
import random
from collections import defaultdict, deque
from mcts import MCTSPlayer
from model import PolicyValueNet
from rdkit.Chem import QED
from rdkit.Chem import AllChem as Chem
import sys


def get_fingerprint(act):
    act=Chem.MolFromSmiles(act)
    return list(Chem.GetMorganFingerprintAsBitVect(act, 2, nBits=1024))

def start_self_play(player, mol, temp=1e-3):
    """Runs a single step within an episode."""

    environment = Molecule(
      ["C", "O", "N"],
      init_mol = mol,
      allow_removal = True,
      allow_no_modification = False,
      allow_bonds_between_rings = False,
      allowed_ring_sizes = [5, 6],
      max_steps = 6,
      target_fn = None,
      record_path = True)
    environment.initialize()
    environment.init_qed=QED.qed(Chem.MolFromSmiles(mol))

    moves, fps, _Qs = player.get_action(environment,
                                         temp=temp,
                                         return_prob=1)

    return zip(fps, _Qs)



class TrainPipeline():
    def __init__(self, mol=None,init_model=None):
        # params of the board and the game
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 20  # num of simulations for each move
        self.c_puct = 1 
        self.buffer_size = 10000
        self.batch_size = 100  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 25  # num of train_steps for each update
        self.kl_targ = 0.2
        self.check_freq = 4
        self.mol=mol
        self.play_batch_size=1
        self.game_batch_num = 20
        self.in_dim=1024
        self.n_hidden_1 = 1024
        self.n_hidden_2 = 1024
        self.out_dim = 1
        self.output_smi=[]
        self.output_qed=[]
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.in_dim,
                                                   self.n_hidden_1,
                                                   self.n_hidden_2,
                                                   self.out_dim,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.in_dim,
                                                   self.n_hidden_1,
                                                   self.n_hidden_2,
                                                   self.out_dim)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)


    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            play_data = start_self_play(self.mcts_player,self.mol,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            print(self.episode_len)
            # augment the data
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        old_probs = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            # mini_batch = random.sample(self.data_buffer, self.batch_size)
            # state_batch = [data[0] for data in mini_batch]
            # mcts_probs_batch = [data[1] for data in mini_batch]
            # old_probs = self.policy_value_net.policy_value(state_batch)
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)))
            )
            #if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
            #    print("early stopping!!")
            #    break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        ))
        return loss, entropy

    def policy_evaluate(self):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        player = MCTSPlayer(self.policy_value_net.policy_value,
                                         c_puct=self.c_puct,
                                         n_playout=20)
        environment = Molecule(
            ["C", "O", "N"],
            init_mol=self.mol,
            allow_removal=True,
            allow_no_modification=False,
            allow_bonds_between_rings=False,
            allowed_ring_sizes=[5, 6],
            max_steps=6,
            target_fn=None,
            record_path=False)
        environment.initialize()
        environment.init_qed = QED.qed(Chem.MolFromSmiles(self.mol))

        moves, fps, _Qs = player.get_action(environment,
                                            temp=self.temp,
                                            return_prob=1,
                                            rand=True)


        return moves


    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i: {}, episode_len: {}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print("loss is {}  entropy is {}".format(loss, entropy))
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    move_list = self.policy_evaluate()
                    # self.policy_value_net.save_model('./current_policy.model')
                    print(move_list)
                    self.output_smi.extend(move_list)
                    o_qed=list(map(lambda x:QED.qed(Chem.MolFromSmiles(x)), move_list))
                    print(o_qed)
                    self.output_qed.extend(o_qed)
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    # training_pipeline = TrainPipeline(mol="O=C(CCC1CCN(c2ncnc3[nH]ccc23)CC1)NCc1ccc(F)cc1")
    training_pipeline = TrainPipeline(mol="OCc1cccc(C[C@@H]2CCN(c3ncnc4[nH]ccc34)C2)c1")
    # training_pipeline = TrainPipeline(mol="C#CNN=O")
    print('result{}.csv'.format(sys.argv[1]))

    training_pipeline.run()
    re={}
    re["Ligand SMILES"] = training_pipeline.output_smi
    re["QED"] = training_pipeline.output_qed
    dataframe=pd.DataFrame.from_dict(re).drop_duplicates(subset=["Ligand SMILES"])
    dataframe.to_csv('result{}.csv'.format(sys.argv[1]),index=False)
