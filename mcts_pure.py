# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes
@author: Junxiao Song
"""

import numpy as np
import copy
import json
from rdkit.Chem import QED
from rdkit.Chem import AllChem as Chem
import random

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def get_fingerprint(act):
    act=Chem.MolFromSmiles(act)
    return list(Chem.GetMorganFingerprintAsBitVect(act, 2, nBits=1024))

root_fp=get_fingerprint("OCc1cccc(C[C@@H]2CCN(c3ncnc4[nH]ccc34)C2)c1")

class TreeNode(object):
    """A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct, n):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        if n<2000:
            return random.sample(self._children.items(), 1)[0]
        else:
            v = list(self._children.items())
            probs=np.array(softmax(list(map(lambda act_node: act_node[1].get_value(c_puct), v))))
            move = np.random.choice(
                probs.size,
                p=0.45*probs + 0.55*np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            return v[move]
        # if not rand:
        #     if np.random.choice(4, 1)[0] == 3:
        #         return random.sample(self._children.items(), 1)[0]
        #     else:
        #         return max(self._children.items(),
        #                   key=lambda act_node: act_node[1]._P)
        # else:
        #     return max(self._children.items(),
        #                key=lambda act_node: act_node[1]._P)

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += leaf_value
        return self._Q

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update(leaf_value*0.8)

        # self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        return self._P+self._Q / (self._n_visits+1)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=1, n_playout=12):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state, n):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root

        while state._counter < state.max_steps:
            if node.is_leaf():
                action_probs = [(state._valid_actions[i],
                                 QED.qed(Chem.MolFromSmiles(state._valid_actions[i])),
                                 )
                                for i in range(len(state._valid_actions_fp))]
                # Check for end of game.
                # print(state._counter, state.max_steps)
                node.expand(action_probs)

                # Greedily select next move.
            action, node = node.select(self._c_puct, n)
            state.step(action)
            # self.update_with_move(action, node._fp)
            if state._counter == state.max_steps:
                # qv=QED.qed(Chem.MolFromSmiles(action))
                node._Q=node._P
                print("###")
            node.update_recursive(node._Q)

        # Update value and visit count of nodes in this traversal.

    def get_move_probs(self, state, rand=False):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            print(n)
            # print("play out {}".format(n))
            state_copy = copy.deepcopy(state)
            self._playout(state_copy, n)

        act_visits=[]
        quen=[(state.init_mol,self._root)]
        # calc the move probabilities based on visit counts at the root node
        while quen:
            act,node = quen.pop(0)
            act_visits.append((act, node._Q))
            if node._children.items():
                act,node=max(node._children.items(),
                    key=lambda act_node: act_node[1]._n_visits)
                quen.append((act,node))

        return act_visits

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPUREPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=10, is_selfplay=1):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    # def get_action(self, environment, temp=1e-3, return_prob=1):
    #     print("#@##########")
    #     # the pi vector returned by MCTS as in the alphaGo Zero paper
    #     # if environment._counter < environment.max_steps:
    #     acts, fps, _Qs = self.mcts.get_move_probs(environment, temp)
    #     print(sum(_Qs))
    #     if self._is_selfplay:
    #         # add Dirichlet Noise for exploration (needed for
    #         # self-play training)
    #         probs= softmax(_Qs)
    #         move = np.random.choice(
    #             len(probs),
    #             p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
    #         )
    #         # update the root node and reuse the search tree
    #         self.mcts.update_with_move(acts[move], fps[move])
    #
    #     if return_prob:
    #         return acts[move], fps, _Qs
    #     else:
    #         return acts[move], fps
        # else:
        #     print("WARNING: the mol is full")
    def get_action(self, environment, temp=1e-3, return_prob=1, rand=False):
        print("#@##########")
        self.reset_player()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        # if environment._counter < environment.max_steps:
        acts_Qs = self.mcts.get_move_probs(environment, rand)

        if return_prob:
            return acts_Qs
        else:
            return acts_Qs
        # else:
        #     print("WARNING: the mol is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
