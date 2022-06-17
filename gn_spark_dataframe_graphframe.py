import sys
import time

from graphframes import GraphFrame
from pyspark import SparkContext, SQLContext, Row
# os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")


def generate_graph(raw_rdd, filter_threshold):
    ub_dict = raw_rdd.groupByKey().mapValues(set).collectAsMap()

    unique_users = raw_rdd.map(lambda s: s[0]).distinct()
    final_rdd = unique_users.cartesian(unique_users)
    print('final_rdd: ', final_rdd.collect())
    final_rdd = final_rdd.map(lambda s: (s[0], s[1], len(ub_dict[s[0]] & ub_dict[s[1]])))\
        .filter(lambda s: s[2] >= filter_threshold and s[0] != s[1])

    edges = final_rdd.map(lambda s: (s[0], s[1])).collect()
    vertices = final_rdd.map(lambda s: s[0]).distinct().map(lambda s: Row(id=s)).collect()

    # print('ub_dict:', ub_dict)
    # print('unique_users: ', unique_users.collect())
    # print('final_rdd: ', final_rdd.collect())
    # print('vertices: ', vertices)
    # print('edges: ', edges)
    return vertices, edges


def execute_task1():
    if len(sys.argv) > 2:
        # filter_threshold, input_file_path, community_output_file_path = sys.argv[1:]
        filter_threshold = int(sys.argv[1])
        input_file_path = sys.argv[2]
        community_output_file_path = sys.argv[3]
    else:
        filter_threshold = 7    # 7
        input_file_path = './data/ub_sample_data.csv'
        # input_file_path = './data/ub_sample_data_small.csv'
        community_output_file_path = './output/output_task1.txt'

    sc = SparkContext('local[*]', 'Task 1')
    sql_context = SQLContext(sc)
    sc.setLogLevel("OFF")   # remove logs
    raw_rdd = sc.textFile(input_file_path)
    first_line = raw_rdd.first()    # remove first line
    raw_rdd = raw_rdd.filter(lambda s: s != first_line).map(lambda s: s.split(","))

    # 1. generate graph
    vertices, edges = generate_graph(raw_rdd, filter_threshold)

    # 2. find communities
    vertices = sql_context.createDataFrame(vertices, ['id'])
    edges = sql_context.createDataFrame(edges, ["src", "dst"])
    g = GraphFrame(vertices, edges)

    result = g.labelPropagation(maxIter=5)
    result_rdd = result.rdd.map(lambda id_label: (id_label[1], id_label[0]))\
        .groupByKey().map(lambda s: sorted(s[1])).sortBy(lambda s: (len(s), s[0])).collect()

    with open(community_output_file_path, "w") as f:
        for user_id in result_rdd:
            f.write(str(user_id)[1:-1] + "\n")


if __name__ == '__main__':
    start_time = time.time()
    execute_task1()
    print('Execution time: ', time.time() - start_time)
