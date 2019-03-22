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
    return list(Chem.GetMorganFingerprintAsBitVect(act, 2, nBits=256))

def start_self_play(player, mol, temp=1e-3):
    """Runs a single step within an episode."""

    environment = Molecule(
      ["C", "O", "N"],
      init_mol = mol,
      allow_removal = True,
      allow_no_modification = True,
      allow_bonds_between_rings = False,
      allowed_ring_sizes = [5, 6],
      max_steps = 10,
      target_fn = None,
      record_path = True)
    environment.initialize()
    environment.init_qed=QED.qed(Chem.MolFromSmiles(mol))
    states, Q = [], []
    for i in range(10):
        qed_l=[QED.qed(Chem.MolFromSmiles(mol)) for mol in environment._valid_actions] 
        ind = np.argmax(qed_l)
        print(qed_l[ind])
        environment.step(environment._valid_actions[ind])
start_self_play(None,"OCc1cccc(C[C@@H]2CCN(c3ncnc4[nH]ccc34)C2)c1")
