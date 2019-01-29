import rospy
import numpy as np
from yolo_ros.msg import *
from yolo_ros.srv import *

class TreeReader(object):
    def __init__(self, srv_name='get_names'):
        rospy.wait_for_service(srv_name)
        srv = rospy.ServiceProxy(srv_name, GetNames)
        res = srv()
        self.names = res.names
        self._nodes = res.tree_nodes
        self._dict = {k:v for v,k in enumerate(self.names)}
        self._roots = [i for i,n in enumerate(self._nodes) if n.parent<0]
        self._leaves = [i for i,n in enumerate(self._nodes) if not n.children]

    def index(self, name):
        if isinstance(name, int):
            return name
        return self._dict[name]

    def parent_index(self, name):
        return self._nodes[self.index(name)].parent

    def parent(self, name):
        ind = self.parent_index(name)
        return self.names[ind] if ind >= 0 else None

    def children_indices(self, name):
        return self._nodes[self.index(name)].children

    def children(self, name):
        return [self.names[ind] for ind in self.children_indices(name)]

    def roots_indices(self):
        roots = filter(lambda x: x[1].parent<0, enumerate(self._nodes))
        return [i for i,_ in roots]

    def roots(self):
        roots = filter(lambda x: x[1].parent<0, enumerate(self._nodes))
        return [self.names[i] for i,_ in roots]

    def levels(self):
        lev = [0] * len(self.names)

        for i in xrange(len(self.names)):
            c = i
            while True:
                parent = self._nodes[c].parent
                if parent < 0:
                    break
                lev[i] += 1
                c = parent
        return lev

    def probability(self, probs, name):
        ind = self.index(name)
        p = probs[ind]
        while True:
            parent = self.parent_index(ind)
            if parent < 0:
                break
            p *= probs[parent]
            ind = parent
        return p

    def trace_max(self, probs):
        inds = self._roots
        trace = []
        prob = 1.
        while inds:
            max_ind = inds[np.argmax([probs[i] for i in inds])]
            prob *= probs[max_ind]
            trace.append((max_ind, self.names[max_ind], prob))
            inds = self.children_indices(max_ind)
        return trace

    def remove(self, probs, ind):
        if isinstance(ind, str):
            ind = self.index(ind)
        probs[ind] = 0
        while True:
            parent = self.parent_index(ind)
            if parent < 0:
                break
            s = 0
            for i in self.children_indices(parent):
                s += probs[i]
            for i in self.children_indices(parent):
                probs[i] /= s
            probs[parent] /= s
            ind = parent
        s = 0
        for i in self.roots_indices():
            s += probs[i]
        for i in self.roots_indices():
            probs[i] /= s

    #conditional to joint
    def to_joint(self, probs):
        pass

    #joint to conditional
    def from_joint(self, probs):
        pass

    def joint_to_leaves(self, probs):
        return [probs[l] for l in self._leaves]

    def joint_from_leaves(self, leaves):
        pass
