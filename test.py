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


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def get_fingerprint(act):
    act=Chem.MolFromSmiles(act)
    return list(Chem.GetMorganFingerprintAsBitVect(act, 2, nBits=256))

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
    states, Q = [], []
    for i in range(100):
        print("###")
        for i in range(6):
            qed_l=[QED.qed(Chem.MolFromSmiles(mol)) for mol in environment._valid_actions]
            print(qed_l)
            for k in qed_l:
                if k > 0.8091:
                    print(k)
                    exit()
                if k> 0.809:
                    print(k)

            probs=softmax(qed_l)
            ind = np.random.choice(
                len(probs),
                p=probs)

            # ind = np.random.choice(len(qed_l), 1)[0]
            # ind = np.argmax(qed_l)

            environment.step(environment._valid_actions[ind])
        environment.initialize()

    print("done!! nothing found")
start_self_play(None,"c1cc(-c2ccc3[nH]ccc3c2)cc(-c2cn[nH]c2)n1")
