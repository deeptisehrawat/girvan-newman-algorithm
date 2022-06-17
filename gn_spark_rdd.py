import sys
import time
from collections import defaultdict
from copy import deepcopy
from pyspark import SparkContext


def generate_graph(raw_rdd, filter_threshold):
    ub_dict = raw_rdd.groupByKey().mapValues(set).collectAsMap()
    unique_users = raw_rdd.map(lambda s: s[0]).distinct()
    final_rdd = unique_users.cartesian(unique_users)
    final_rdd = final_rdd.map(lambda s: (s[0], s[1], len(ub_dict[s[0]] & ub_dict[s[1]])))\
        .filter(lambda s: s[2] >= filter_threshold and s[0] != s[1])

    edges = final_rdd.map(lambda s: (s[0], s[1])).collect()
    vertices = final_rdd.map(lambda s: s[0]).distinct().collect()

    graph = defaultdict(set)
    for first, second in edges:
        graph[first].add(second)
        graph[second].add(first)

    return vertices, edges, graph


def bfs(root, graph):
    visited = set()
    parent_dict = defaultdict(set)
    level_dict = dict()
    num_shortest_paths = defaultdict(int)
    traversal = []

    queue = [root]
    visited.add(root)
    level_dict[root] = 0
    num_shortest_paths[root] = 1

    while queue:
        root = queue.pop(0)
        traversal.append(root)

        for node in graph[root]:
            if node not in visited:
                queue.append(node)
                visited.add(node)
                parent_dict[node].add(root)
                num_shortest_paths[node] += num_shortest_paths[root]
                level_dict[node] = level_dict[root] + 1
            # node is on next level, hence child of root
            elif level_dict[node] == level_dict[root] + 1:
                parent_dict[node].add(root)
                num_shortest_paths[node] += num_shortest_paths[root]

    # print('traversal: ', traversal)
    # print('parent_dict: ', parent_dict)
    # print('level_dict: ', level_dict)
    return parent_dict, traversal, num_shortest_paths


def get_credits(parent_dict, traversal, num_shortest_paths):
    node_credits = dict()
    edge_credits = defaultdict(float)
    for node in traversal:
        node_credits[node] = 1

    traversal.reverse()
    for node in traversal:
        for parent in parent_dict[node]:
            value = node_credits[node] * float(num_shortest_paths[parent] / num_shortest_paths[node])
            node_credits[parent] += value
            edge_credits[tuple(sorted([node, parent]))] += value

    return edge_credits


def girvan_newman(graph, vertices):
    betweenness = defaultdict(float)
    for vertex in vertices:
        # 1. run bfs to determine parent relationships for step 3 (bottom-up)
        # 2. number of shortest paths
        parent_dict, traversal, num_shortest_paths = bfs(vertex, graph)
        # 3. find edge and node credits
        edge_credits = get_credits(parent_dict, traversal, num_shortest_paths)
        # print('edge_credits: ', edge_credits)
        # 4. sum for all edges and divide by two
        for k, v in edge_credits.items():
            # print('k: ', k, ' v: ', v)
            betweenness[k] += float(v / 2)

    # sort dict by value (desc) and first user_id
    return sorted(betweenness.items(), key=lambda s: (-s[1], s[0]))


def get_connected_components(edit_graph, vertices):
    connected_components = []
    # visit all vertices once to get all connected components
    while len(vertices) > 0:
        # bfs: to find connected component from a vertex
        root = vertices.pop()
        visited = set()
        queue = [root]
        visited.add(root)
        while queue:
            root = queue.pop(0)
            for node in edit_graph[root]:
                if node not in visited:
                    queue.append(node)
                    visited.add(node)
                    vertices.remove(node)
        # add connected component to communities
        connected_components.append(sorted(visited))

    return connected_components


def get_curr_modularity(graph, m, curr_communities):
    modularity = 0
    for s in curr_communities:
        for i in s:
            for j in s:
                is_connected_ij = 0
                if j in graph[i]:
                    is_connected_ij = 1
                modularity += is_connected_ij - ((len(graph[i]) * len(graph[j])) / (2 * m))

    modularity /= (2 * m)
    return modularity


def find_communities(graph, vertices, betweenness):
    communities = []
    edit_graph = deepcopy(graph)
    m = len(betweenness)    # number of edges in original graph
    highest_modularity = -1

    while betweenness:
        # list of connected components
        connected_components = get_connected_components(edit_graph, deepcopy(vertices))
        curr_modularity = get_curr_modularity(graph, m, connected_components)
        if curr_modularity > highest_modularity:
            highest_modularity = curr_modularity
            # communities = connected_components.copy()
            communities = deepcopy(connected_components)

        # remove edges with highest betweenness
        highest_betweenness = betweenness[0][1]
        for (node_x, node_y), bw in betweenness:
            if bw < highest_betweenness:
                break
            # remove this edge
            edit_graph[node_x].remove(node_y)
            edit_graph[node_y].remove(node_x)

        # re-calculate betweenness
        betweenness = girvan_newman(edit_graph, vertices)

    communities.sort(key=lambda s: (len(s), s[0]))
    return communities


def execute_task2():
    if len(sys.argv) > 2:
        filter_threshold = int(sys.argv[1])
        input_file_path, betweenness_output_file_path, community_output_file_path = sys.argv[2:]
    else:
        filter_threshold = 7    # 7
        # input_file_path = './data/ub_sample_data.csv'
        input_file_path = './test_data/test_user_business.csv'
        # input_file_path = './data/ub_sample_data_small.csv'
        betweenness_output_file_path = './output/output_task2_1.txt'
        community_output_file_path = './output/output_task2_2.txt'

    sc = SparkContext('local[*]', 'Task 2')
    sc.setLogLevel("OFF")   # remove logs
    raw_rdd = sc.textFile(input_file_path)
    first_line = raw_rdd.first()    # remove first line
    raw_rdd = raw_rdd.filter(lambda s: s != first_line).map(lambda s: s.split(","))

    # 1. generate graph
    vertices, edges, graph = generate_graph(raw_rdd, filter_threshold)
    # print('vertices: ', vertices)
    # print('edges: ', edges)
    # print('graph: ', graph)

    # graph = {
    #     'A': {'B', 'C'},
    #     'B': {'A', 'C', 'D'},
    #     'C': {'A', 'B'},
    #     'D': {'B', 'E', 'F', 'G'},
    #     'E': {'D', 'F'},
    #     'F': {'D', 'E', 'G'},
    #     'G': {'D', 'F'},
    # }
    # bfs('E', graph)

    # 2. girvan newman to find betweenness
    betweenness = girvan_newman(graph, vertices)
    # print('betweenness: ', betweenness)

    # write betweenness to output file
    with open(betweenness_output_file_path, "w") as f:
        for i in betweenness:
            f.write(str(i[0]) + "," + str(round(i[1], 5)) + "\n")

    # 3. use betweenness to find communities
    communities = find_communities(graph, vertices, betweenness)

    # write communities to output file
    with open(community_output_file_path, "w") as f:
        for user_id in communities:
            f.write(str(user_id)[1:-1] + "\n")


if __name__ == '__main__':
    start_time = time.time()
    execute_task2()
    print('Execution time: ', time.time() - start_time)
