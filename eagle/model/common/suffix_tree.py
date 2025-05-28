import numpy as np
from collections import defaultdict, deque


class SAMNode:
    def __init__(self):
        self.next = {}  # token -> state
        self.link = -1
        self.length = 0
        self.freq = 0  # 终止频率（该状态对应子串的出现次数）


class FreqSuffixAutomaton:
    def __init__(self):
        self.nodes = [SAMNode()]
        self.last = 0

    def extend(self, token):
        cur = len(self.nodes)
        self.nodes.append(SAMNode())
        self.nodes[cur].length = self.nodes[self.last].length + 1
        p = self.last
        while p != -1 and token not in self.nodes[p].next:
            self.nodes[p].next[token] = cur
            p = self.nodes[p].link
        if p == -1:
            self.nodes[cur].link = 0
        else:
            q = self.nodes[p].next[token]
            if self.nodes[p].length + 1 == self.nodes[q].length:
                self.nodes[cur].link = q
            else:
                clone = len(self.nodes)
                self.nodes.append(SAMNode())
                self.nodes[clone].length = self.nodes[p].length + 1
                self.nodes[clone].next = self.nodes[q].next.copy()
                self.nodes[clone].link = self.nodes[q].link
                while p != -1 and self.nodes[p].next.get(token, -1) == q:
                    self.nodes[p].next[token] = clone
                    p = self.nodes[p].link
                self.nodes[q].link = clone
                self.nodes[cur].link = clone
        self.last = cur
        return cur

    def build(self, sequences):
        """
        sequences: List of token lists, 多条序列
        """
        self.nodes = [SAMNode()]
        self.last = 0

        for seq in sequences:
            self.last = 0
            for t in seq:
                cur = self.extend(t)
                self.nodes[cur].freq += 1  # 每个状态频率+1，统计所有出现子串频率

        self._propagate_freq()

    def _propagate_freq(self):
        """
        拓扑排序逆序传播频率：
        频率从长子串传递到后缀链接，保证父状态频率包含所有子状态频率
        """
        max_len = max(node.length for node in self.nodes)
        buckets = [[] for _ in range(max_len + 1)]
        for i, node in enumerate(self.nodes):
            buckets[node.length].append(i)
        for length in range(max_len, 0, -1):
            for idx in buckets[length]:
                link = self.nodes[idx].link
                if link != -1:
                    self.nodes[link].freq += self.nodes[idx].freq

    def next_tokens_prob(self, query_tokens):
        """
        查询 query_tokens 对应状态的后续token及概率，并返回对应子树状态索引
        返回: List of (token, prob, next_state)
        """
        current_state = 0
        for t in query_tokens:
            if t not in self.nodes[current_state].next:
                return []  # 子串不存在
            current_state = self.nodes[current_state].next[t]

        total_weight = 0
        scores = {}
        for token, nxt_state in self.nodes[current_state].next.items():
            freq = self.nodes[nxt_state].freq
            score = freq
            scores[token] = (score, nxt_state)
            total_weight += score

        if total_weight == 0:
            return []

        # 归一化成概率，并组合 next_state
        result = [
            (token, score / total_weight, nxt_state)
            for token, (score, nxt_state) in scores.items()
        ]
        # 按概率倒序排列
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_all_paths(self, current_state, cur_token, max_depth=10):
        """
        1. 遍历子树，生成 state_list（所有节点状态编号，访问顺序，不重复）
        2. 生成 token_list，token_list[i] 是 state_list[i] 对应节点的进入token（根节点token为None）
        3. 生成所有路径，用 state_list 索引表示路径（起点是 current_state）

        返回:
          state_list: List[int]  # 节点状态编号列表
          token_list: List[token or None]  # 进入该状态的token，根节点None
          paths: List[List[int]]  # 路径由 state_list 索引组成
        """
        state_list = []
        token_list = []
        state_to_idx = {}

        def dfs_collect(state, parent=None, token_from_parent=None):
            if state not in state_to_idx:
                state_to_idx[state] = len(state_list)
                state_list.append(state)
                token_list.append(token_from_parent)  # 根节点为None
            node = self.nodes[state]
            for token, nxt_state in node.next.items():
                dfs_collect(nxt_state, state, token)

        dfs_collect(current_state, None, cur_token)

        paths = []

        def dfs_paths(state_idx, path):
            state = state_list[state_idx]
            node = self.nodes[state]
            if len(path) >= max_depth or len(node.next) == 0:
                paths.append(path[:])
                return
            for token, nxt_state in node.next.items():
                nxt_idx = state_to_idx[nxt_state]
                dfs_paths(nxt_idx, path + [nxt_idx])

        start_idx = state_to_idx[current_state]
        dfs_paths(start_idx, [start_idx])

        return state_list, token_list, paths

    def print_subtree(self, start_state, max_depth=10, indent=0):
        """
        递归打印从 start_state 节点开始的子树结构
        max_depth: 限制递归深度，避免无限递归
        indent: 缩进层级，用于美观输出
        """
        if max_depth == 0:
            print(" " * indent + "...")
            return
        node = self.nodes[start_state]
        for token, nxt_state in node.next.items():
            print(
                " " * indent
                + f"Token {token} -> State {nxt_state} (freq={self.nodes[nxt_state].freq})"
            )
            self.print_subtree(nxt_state, max_depth - 1, indent + 4)


def build_tree_attention_mask(num_nodes, paths):
    # 初始化 mask 矩阵，全为0
    mask = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    # 遍历每条路径
    for path in paths:
        # 遍历路径节点，i是关注者
        for i_idx, i_node in enumerate(path):
            # i_node 可以关注路径中 0 到 i_idx 的节点（包括自己和祖先）
            for j_node in path[: i_idx + 1]:
                mask[i_node][j_node] = 1

    return mask


if __name__ == "__main__":
    sam = FreqSuffixAutomaton()
    # 构建样例序列
    sequences = [
        [1, 2, 1, 3, 4, 5, 6],  # 序列A
        [1, 2, 3, 6, 7],  # 序列B
        [3, 4, 5, 8],  # 序列C
        [3, 4, 2],  # 序列D
        [1, 2, 3, 4, 5],  # 序列A 再来一次，增加频率
        [3, 4, 5, 6, 7, 8],  # 序列C 再来一次，增加频率
        [3, 4, 5, 6, 8, 8],  # 序列C 再来一次，增加频率
    ]
    sam.build(sequences)

    # 打印所有节点的freq，方便验证频率传播
    print("节点频率统计:")
    for i, node in enumerate(sam.nodes):
        print(
            f"节点{i}: 长度={node.length}, link={node.link}, freq={node.freq}, next={list(node.next.keys())}"
        )
    # 测试 next_tokens_prob
    query = [3, 4]
    results = sam.next_tokens_prob(query)

    # 可以返回多个子树，以及每个子树的概率
    # TODO：gulihui 输出所有draft token list
    # 输出序列，索引用draft token list中的序列
    # 树状注意力掩码 tree mask
    # 对应draft token 的深度，根节点为0

    print(f"查询序列 {query} 的后续token概率和子树状态:")
    for token, prob, next_state in results:
        print(f"Token: {token}, 概率: {prob:.4f}, 对应子树状态节点:")
        _, draft_tokens, paths = sam.get_all_paths(next_state, token)
        print(draft_tokens)
        print(paths)
        mask = build_tree_attention_mask(len(draft_tokens), paths)
        print(mask)
