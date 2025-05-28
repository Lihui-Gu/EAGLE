import json
import pickle
from collections import defaultdict

import numpy as np


class Node:
    __slots__ = ["freqs", "children"]

    def __init__(self, children, freqs):
        self.children = children
        self.freqs = freqs

    def __repr__(self):
        return f"{list(self.children.keys())}:{self.freqs}"


class Tree:
    def __init__(self, token_id, max_node=65536, max_output_node=512):
        self.token_id = token_id
        self.max_node = max_node
        self.max_output_node = max_output_node
        self.n_node = 0
        self.n_output_node = 0
        self.nodes = {}

    def put(self, token_ids, mode="output", idx=0, freq=1.0):
        assert mode in ("input", "output")
        if mode == "output":
            idx = -1
        self._put(token_ids, self.nodes, mode=mode, idx=idx, freq=freq)

    def _put(self, token_ids, nodes, mode="output", freq=1.0, idx=-1):
        # 递归调用
        while True:
            if len(token_ids) == 0:
                break

            t = token_ids[0] # 取出当前序列的第一个token
            node = nodes.get(t, None) # 查看当前nodes中是否已经有该节点了

            if node is None:
                # 如果当前nodes中没有就需要构建路径
                n = self._pack(token_ids, idx, freq=freq)
                nodes.update(n) # 新节点插入子树中
                self.n_node += len(token_ids)
                if mode == "output":
                    self.n_output_node += len(token_ids)
                break

            node.freqs[idx] = node.freqs.get(idx, 0.0) + freq
            nodes = node.children
            token_ids = token_ids[1:]

    def _pack(self, token_ids, idx, freq=1.0):
        ps = {}
        for token in token_ids[::-1]:
            freqs = {idx: freq}
            p = Node(ps, freqs)
            ps = {token: p}
        return ps

    def get(
        self,
        token_ids,
        max_size=64,
        max_length=8,
        min_input_size=0,
        min_output_size=0,
        output_weight=1e-4,
        mode="mix",
        idx=0,
    ):
        assert mode in ("input", "output", "mix")

        match_token_id, nodes = self._match(token_ids, mode=mode, idx=idx)
        if len(nodes) == 0:
            token_id = token_ids[-1] if len(token_ids) > 0 else self.token_id
            return [token_id], np.ones((1, 1), dtype=np.int64), [0, 0]

        freqs = []
        self._dfs_get_freqs(nodes, freqs, idx, output_weight)
        # self._bfs_get_freqs(nodes, freqs, idx, output_weight)

        min_mix_freq = 1e9
        min_input_freq = 1e9
        min_output_freq = 1e9
        if mode == "input":
            output_weight = 0.0
            size = len([x for x in freqs if x[1] > 0])
            if size > max_size:
                input_freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
                min_input_freq = input_freqs[min_input_size - 1][1]
            else:
                min_input_freq = 0.0
        elif mode == "output":
            output_weight = 1.0
            size = len([x for x in freqs if x[2] > 0])
            if size > max_size:
                output_freqs = sorted(freqs, key=lambda x: x[2], reverse=True)
                min_output_freq = output_freqs[min_output_size - 1][2]
            else:
                min_output_freq = 0.0
        else:
            size = len([x for x in freqs if x[1] > 0 or x[2] > 0])
            if size > max_size:
                indices = set()
                if min_input_size > 0:
                    input_freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
                    min_input_freq = input_freqs[min_input_size - 1][1]
                    indices.update([x[0] for x in input_freqs[:min_input_size]])

                if min_output_size > 0:
                    output_freqs = sorted(freqs, key=lambda x: x[2], reverse=True)
                    min_output_freq = output_freqs[min_output_size - 1][2]
                    indices.update([x[0] for x in output_freqs[:min_output_size]])

                if len(indices) < max_size:
                    mix_freqs = sorted(freqs, key=lambda x: x[3], reverse=True)
                    rest_size = max_size - len(indices)
                    indices.update([x[0] for x in mix_freqs[:rest_size]])
                    cur_size = len(indices)
                    for i in range(rest_size, min(rest_size + max_size, size)):
                        if mix_freqs[i][0] in indices:
                            continue
                        cur_size += 1
                        if cur_size >= max_size:
                            x = mix_freqs[i]
                            min_mix_freq = x[3]
                            break
            else:
                min_mix_freq = 0.0

        mask = np.zeros((max_size, max_size), dtype=np.int64)
        mask[:, 0] = 1
        ids = [match_token_id or self.token_id]
        sizes = [0, 0]
        self._ravel(
            nodes,
            ids,
            mask,
            -1,
            max_size=max_size,
            max_length=max_length,
            min_output_freq=min_output_freq,
            min_input_freq=min_input_freq,
            min_mix_freq=min_mix_freq,
            sizes=sizes,
            output_weight=output_weight,
            mode=mode,
            idx=idx,
        )
        size = len(ids)

        mask = mask[:size, :size]
        return ids, mask, sizes

    def _dfs_get_freqs(self, nodes, freqs, idx, output_weight):
        for node in nodes.values():
            fo = node.freqs.get(-1, 0.0)
            fi = node.freqs.get(idx, 0.0)
            if fo > 0 or fi > 0:
                fm = (1.0 - output_weight) * fi + output_weight * fo
                freqs.append([None, fi, fo, fm])
                if len(node.children) > 0:
                    self._dfs_get_freqs(node.children, freqs, idx, output_weight)

    def _bfs_get_freqs(self, nodes, freqs, idx, output_weight):
        node_list = [nodes]
        while len(node_list) > 0:
            update_node_list = []
            for nodes in node_list:
                for node in nodes.values():
                    fo = node.freqs.get(-1, 0.0)
                    fi = node.freqs.get(idx, 0.0)
                    if fo > 0 or fi > 0:
                        fm = (1.0 - output_weight) * fi + output_weight * fo
                        freqs.append([None, fi, fo, fm])
                        if len(node.children) > 0:
                            update_node_list.append(node.children)
            node_list = update_node_list

    def get_one_branch(self, token_ids, max_length=8, mode="mix", idx=0):
        assert mode in ("input", "output", "mix")

        match_token_id, nodes = self._match(token_ids, mode=mode, idx=idx)
        if len(nodes) == 0:
            token_id = token_ids[-1] if len(token_ids) > 0 else self.token_id
            return [token_id], np.ones((1, 1), dtype=np.int64), [0, 0]

        ids = [match_token_id or self.token_id]
        length = 0
        while True:
            if len(nodes) == 0 or length >= max_length:
                break
            max_freq = 0.0
            max_node = None
            max_id = None
            if mode == "mix":
                for t, node in nodes.items():
                    freqs = node.freqs
                    fo = freqs.get(idx, 0.0)
                    fi = freqs.get(-1, 0.0)
                    if fo > 0 or fi > 0:
                        freq = 10000 * fi + fo
                        if freq > max_freq:
                            max_freq = freq
                            max_node = node
                            max_id = t
            elif mode == "input":
                for t, node in nodes.items():
                    freqs = node.freqs
                    freq = freqs.get(idx, 0.0)
                    if freq > 0:
                        if freq > max_freq:
                            max_freq = freq
                            max_node = node
                            max_id = t
            else:
                for t, node in nodes.items():
                    freqs = node.freqs
                    freq = freqs.get(-1, 0.0)
                    if freq > 0:
                        if freq > max_freq:
                            max_freq = freq
                            max_node = node
                            max_id = t
            if max_node is None:
                break
            ids.append(max_id)
            nodes = max_node.children
            length += 1

        return (
            ids,
            np.tril(np.ones((length + 1, length + 1), dtype=np.int64), 0),
            [length],
        )

    def _match(self, token_ids, mode="mix", idx=0):
        nodes = self.nodes
        token_id = None
        if len(token_ids) == 0:
            return token_id, nodes

        for token_id in token_ids:
            node = nodes.get(token_id, None)
            nodes = {}
            if node is None:
                break

            if mode == "input":
                if node.freqs.get(idx, 0.0) > 0:
                    nodes = node.children
            elif mode == "output":
                if node.freqs.get(-1, 0.0) > 0:
                    nodes = node.children
            else:
                if node.freqs.get(idx, 0.0) > 0 or node.freqs.get(-1, 0.0) > 0:
                    nodes = node.children

        return token_id, nodes

    def _ravel(
        self,
        nodes,
        ids,
        mask,
        pid,
        max_size=64,
        max_length=8,
        min_output_freq=1.0,
        min_input_freq=1.0,
        min_mix_freq=1.0,
        output_weight=1e-4,
        sizes=None,
        mode="mix",
        idx=0,
    ):
        if len(ids) >= max_size or max_length <= 0:
            return

        sorts = [
            (
                k,
                v,
                (1.0 - output_weight) * v.freqs.get(idx, 0.0)
                + output_weight * v.freqs.get(-1, 0.0),
            )
            for k, v in nodes.items()
        ]
        sorts = sorted(sorts, key=lambda x: x[2], reverse=True)
        for tid, node, fm in sorts:
            if len(ids) >= max_size:
                return
            fi = node.freqs.get(idx, 0.0)
            fo = node.freqs.get(-1, 0.0)
            if mode == "mix":
                if fi < min_input_freq and fo < min_output_freq and fm < min_mix_freq:
                    continue
            elif mode == "input":
                if fi < min_input_freq:
                    continue
            else:
                if fo < min_output_freq:
                    continue
            if fi > 0.0:
                sizes[0] += 1
            if fo > 0.0:
                sizes[1] += 1
            ids.append(tid)
            rid = len(ids) - 1

            if pid > -1:
                mask[rid] = mask[pid]
            mask[rid, rid] = 1
            if len(node.children) > 0:
                self._ravel(
                    node.children,
                    ids,
                    mask,
                    rid,
                    max_size=max_size,
                    max_length=max_length - 1,
                    min_output_freq=min_output_freq,
                    min_input_freq=min_input_freq,
                    min_mix_freq=min_mix_freq,
                    output_weight=output_weight,
                    sizes=sizes,
                    mode=mode,
                    idx=idx,
                )

    def squeeze(self):
        if self.n_node > self.max_node or self.n_output_node > self.max_output_node:
            self._squeeze(self.nodes)
            sizes = [0]
            self._count_node(self.nodes, sizes)
            self.n_node = sizes[0]
            self.n_output_node = sizes[0]

    def _squeeze(self, nodes):
        for t, p in list(nodes.items()):
            fo = p.freqs.get(-1, 0.0)
            if fo > 1.0:
                p.freqs[-1] *= 0.5
                if len(p.children) > 0:
                    self._squeeze(p.children)
            else:
                nodes.pop(t)

    def _count_node(self, nodes, sizes):
        l = len(nodes)
        sizes[0] += l
        for t, n in nodes.items():
            if len(n.children) > 0:
                self._count_node(n.children, sizes)

    def reset_input_freq(self, idx):
        if len(self.nodes) == 0:
            return
        self._reset_input_freq(self.nodes, idx)

    def _reset_input_freq(self, nodes, idx):
        for t, node in nodes.items():
            freqs = node.freqs
            f = freqs.get(idx, 0.0)
            if f == 0.0:
                continue
            freqs[idx] = 0.0
            if len(node.children) > 0:
                self._reset_input_freq(node.children, idx)
