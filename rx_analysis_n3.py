# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import *
from pyspark.sql.session import SparkSession

np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))

conf = SparkConf().setAppName("App")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '100G')
        .set('spark.driver.memory', '100G')
        .set('spark.driver.maxResultSize', '100G'))
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

NODE_COUNT = None
PARENT_NODE_BASE = 80


def get_origin_data():
    """
    得到原始数据
    """
    # input_lh_tp_node_ui = sqlContext.read.format("jdbc").options(
        # url="jdbc:hive2://192.168.1.114:10000/zkfx_test?hive.resultset.use.unique.column.names=false",
        # driver="org.apache.hive.jdbc.HiveDriver", dbtable="lh_tp_node_ui", user="admin", password="admin").load()
    input_lh_tp_node_ui = spark.read.csv("lh_tp_node_ui.csv", header=True, inferSchema=True)
    # input_lh_tp_node_ui = spark.read.csv("1000_generated_test_data.csv", header=True, inferSchema=True)
    input_lh_tp_node_ui = input_lh_tp_node_ui.filter('lv == 1')
    return input_lh_tp_node_ui


def get_corr_max_two_node_name(corrs):
    """
    得到电压相关性最大的两个节点的下标
    :param corrs: 矩阵
    :return: 节点下标
    """
    return np.unravel_index(corrs.argmax(), corrs.shape)


def linear_regression(node_x, node_y, node_join_xy, phase='A'):
    """
    两节点在单个相位上做线性回归分析
    :param node_x: node_x name
    :param node_y: node_y name
    :node_join_xy: node_x node_y拼接后的数据
    :param phase: 相位
    :return: 回归系数列表
    """
    # 筛选出单相位的数据
    s3 = 'l1 == "%s"' % phase
    node_join_p = node_join_xy.filter(s3)
    if node_join_p.count() == 0:
        return None
    node_join_p = node_join_p.drop('node1', 'node2', 'l1')
    assembler = VectorAssembler(inputCols=["u1", "ir1", "ix1", "ir2", "ix2"], outputCol="features")
    output = assembler.transform(node_join_p)
    label_features = output.select("features", "u2")#.toDF('features', 'label')
    # regularization?
    lr = LinearRegression(maxIter=5, elasticNetParam=0.8, labelCol="u2")
    fit_begin_time = time.time()
    print('fit begin')
    lrModel = lr.fit(label_features)
    print('fit time', time.time() - fit_begin_time)
    trainingSummary = lrModel.summary
    param = [
        node_x, node_y, phase, trainingSummary.r2,
        lrModel.intercept,
        *lrModel.coefficients,
    ]
    return param


def get_linear_regression_param_list(data, node_x, node_y):
    """
    两节点在A、B、C三个相位上分别做线性回归
    :param data: 原始数据
    :param node_x: node_x name
    :param node_y: node_y name
    :return: 不同相位的回归系数 ~ spark-df/ node_x,node_y合并后的数据
    """
    # 生成做回归分析的数据
    s_x = "node == {}".format(node_x)
    s_y = "node == {}".format(node_y)
    nodex = data.filter(s_x)
    nodey = data.filter(s_y)
    nodex = (nodex.withColumnRenamed('node', 'node1')
             .withColumnRenamed('u', 'u1')
             .withColumnRenamed('ir', 'ir1')
             .withColumnRenamed('ix', 'ix1'))
    nodey = (nodey.withColumnRenamed('node', 'node2')
             .withColumnRenamed('l1', 'l2')
             .withColumnRenamed('u', 'u2')
             .withColumnRenamed( 'ir', 'ir2')
             .withColumnRenamed('ix', 'ix2')
             .withColumnRenamed('data_time', 'data_time2'))
    node_join_xy = nodex.join(nodey, ((nodex['data_time'] == nodey.data_time2) & (nodex['l1'] == nodey.l2)))
    node_join_xy = node_join_xy.select('node1', 'node2', 'data_time', 'l1', 'u1', 'ir1', 'ix1', 'u2', 'ir2', 'ix2')
    node_join_xy = node_join_xy.withColumn("ir2", node_join_xy["ir2"] * (-1))
    node_join_xy = node_join_xy.withColumn("ix2", node_join_xy["ix2"] * (-1))
    node_join_xy = node_join_xy.repartition(80)
    print('node_join_xy')
    node_join_xy.toDF('node1', 'node2', 'data_time', 'l1', 'u1', 'ir1', 'ix1', 'u2', 'ir2', 'ix2').sort(asc('data_time')).show()
    # 获得两表~三相位~的回归系数列表
    param_list = []
    for phase in ['A', 'B', 'C']:
        result = linear_regression(node_x, node_y, node_join_xy, phase=phase)
        if result is not None:
            param_list.append(result)
    name = ['node1', 'node2', 'phase', 'rsquare', 'b0', 'b1', 'r1', 'x1', 'r2', 'x2']
    param_df = pd.DataFrame(columns=name, data=param_list)
    # TODO: if empty
    if param_df.shape[0] == 0:
        param_df = pd.DataFrame(columns=name, data=[[0 for _ in name]])
    param_dfs = spark.createDataFrame(param_df)
    return param_dfs, node_join_xy


def updata_node_couple(data, node_join_xy, param_dfs, node_x, node_y, q):
    """
    更新原始数据：从节点集合中删除两个子节点，添加父节点
    :param data: 原始数据
    :param param_dfs: 两节点的回归系数 ~ dfs
    :return: 删除两个子节点，添加父节点后的输入数据
    """
    # 求出新的虚拟表箱的数据u ir ix
    node_couple_xy = param_dfs.withColumnRenamed("node1", "node3").withColumnRenamed("node2", "node4")
    node_xy = node_join_xy.join(node_couple_xy,
                                ((node_join_xy.node2 == node_couple_xy.node3)
                                 & (node_join_xy.node1 == node_couple_xy.node4)) |
                                ((node_join_xy.node1 == node_couple_xy.node3)
                                 & (node_join_xy.node2 == node_couple_xy.node4)) &
                                (node_join_xy.l1 == node_couple_xy.phase)
                                , 'left')
    node_xy = node_xy.withColumn('u', (node_xy['u1'] + node_xy['u2']
                                       + node_xy['ir1'] * node_xy['r1'] + node_xy['ix1'] * node_xy['x1']
                                       + node_xy['ir2'] * node_xy['r2'] + node_xy['ix2'] * node_xy['x2']) / 2)
    node_xy = node_xy.withColumn('ir', (node_xy['ir1'] + node_xy['ir2']))
    node_xy = node_xy.withColumn('ix', (node_xy['ix1'] + node_xy['ix2']))
    node_xy = node_xy.withColumn('node', lit(PARENT_NODE_BASE + q))
    node_xy = node_xy.withColumn('lv', lit(q + 2))
    node_xy = node_xy.select('lv', 'node', 'l1', 'data_time', 'u', 'ir', 'ix')
    # 更新数据
    s_filter = 'node != "{}" and node != "{}"'.format(node_x, node_y)
    data = data.filter(s_filter)
    new_data = data.union(node_xy)
    # new_data = new_data.coalesce(1)
    return new_data


def get_primary_secondary_single_data(param_dfs, node_x, node_y, q):
    """
    :param param_dfs: 两节点的回归系数
    :return: 主副表单条数据
    """
    # 副表单条数据
    data_x = param_dfs
    s1_dfs = data_x.select('node1', 'phase', 'rsquare', 'r1', 'x1', 'b0', 'b1')
    s2_dfs = data_x.select('node2', 'phase', 'rsquare', 'r2', 'x2', 'b0', 'b1')
    # 主表单条数据
    avg_rsquare = data_x.groupBy("node1", "node2").avg("rsquare").toPandas()['avg(rsquare)'].values[0]
    avg_r1 = data_x.groupBy("node1", "node2").avg("r1").toPandas()['avg(r1)'].values[0]
    avg_r2 = data_x.groupBy("node1", "node2").avg("r2").toPandas()['avg(r2)'].values[0]
    avg_x1 = data_x.groupBy("node1", "node2").avg("x1").toPandas()['avg(x1)'].values[0]
    avg_x2 = data_x.groupBy("node1", "node2").avg("x2").toPandas()['avg(x2)'].values[0]
    avg_b0 = data_x.groupBy("node1", "node2").avg("b0").toPandas()['avg(b0)'].values[0]
    avg_b1 = data_x.groupBy("node1", "node2").avg("b1").toPandas()['avg(b1)'].values[0]
    avg_z1 = (avg_r1 ** 2 + avg_x1 ** 2) ** 0.5
    avg_z2 = (avg_r2 ** 2 + avg_x2 ** 2) ** 0.5
    p1_list = [node_x, PARENT_NODE_BASE + q, avg_rsquare, avg_r1, avg_x1, avg_z1, avg_b0, avg_b1]
    p2_list = [node_y, PARENT_NODE_BASE + q, avg_rsquare, avg_r2, avg_x2, avg_z2, avg_b0, avg_b1]
    return s1_dfs, s2_dfs, p1_list, p2_list


def save_table_data(primary_list, secondary_df):
    # TODO: df.write.format("csv").save(filepath)
    # 主表存储
    name = ['NODE_ID', 'PARENT_NODE_ID', 'RSQUARED', 'R', 'X', 'Z', 'B0', 'B1']
    primary_table = pd.DataFrame(data=primary_list, columns=name).reset_index().rename(columns={'index': 'DATA_ID'})
    primary_table.to_csv('new_4_primary_table.csv')
    # 副表存储
    secondary_list = np.array(secondary_df.collect()).tolist()
    name = ['NODE', 'PHASE', 'RSQUARED', 'R', 'X', 'B0', 'B1']
    secondary_tmp = pd.DataFrame(data=secondary_list, columns=name)[1:]
    join_data = primary_table[['DATA_ID', 'NODE_ID']]
    secondary_table = pd.merge(secondary_tmp, join_data, how='left', left_on='NODE', right_on='NODE_ID')
    secondary_table = secondary_table.reset_index().rename(
        columns={'DATA_ID': 'RESULT_DATA_ID', 'index': 'DATA_ID'}).drop(
        ['NODE_ID', 'NODE'], axis=1)
    secondary_table['R'] = secondary_table['R'].astype(float)
    secondary_table['X'] = secondary_table['X'].astype(float)
    secondary_table['Z'] = (secondary_table['R'] ** 2 + secondary_table['X'] ** 2) ** 0.5
    secondary_table.to_csv('new_4_secondary_table.csv')
    return primary_table, secondary_table


def get_corr_matrix(input_lh_tp_node_ui, node_names):
    # 假设节点列表可以存进master内存
    # 分布式矩阵：https://stackoverflow.com/questions/33558755/matrix-multiplication-in-apache-spark
    header=['DATA_TIME', 'L1', 'U']

    # key: node
    # value: pd.DataFrame
    pairRDD = (input_lh_tp_node_ui.rdd
               .map(lambda r: (r['NODE'], (r['DATA_TIME'], r['L1'], r['U'])))
               .groupByKey())
    node_names_2_idx = { o: i for i, o in enumerate(node_names) }
    pairRDD = pairRDD.map(lambda x: (node_names_2_idx[x[0]], pd.DataFrame(x[1].data, columns=header)))

    def _rdd_struct_corr(structs):
        (node1, df1), (node2, df2) = structs
        # 只计算上三角
        if node1 >= node2:
            return node1, node2, -np.inf

        df_join = df1.merge(df2, on=('DATA_TIME', 'L1'), suffixes=('_l', '_r'))
        # TODO: coalesce?
        corr = df_join[['U_l', 'U_r']].corr().iloc[0,1]
        return node1, node2, corr
    cartesian = pairRDD.cartesian(pairRDD)
    corrs_rdd = cartesian.map(_rdd_struct_corr).sortBy(lambda r: (r[0], r[1]))
    corrs = np.array(corrs_rdd.map(lambda r: r[2]).collect()).reshape((NODE_COUNT, NODE_COUNT))
    return pairRDD, corrs


def update_distance(pairRDD, node_join_xy, param_dfs, idx_node_x, idx_node_y, corrs, q, inactive_set):
    """
    参数：
        node_names: 修改内存
        corrs: 修改内存
        inactive_set: 储存被合并的节点index集合，初始化为空集合，修改内存
        pairRDD: 返回
    """
    # 创建df_parent
    # 求出新的虚拟表箱的数据u ir ix
    node_couple_xy = param_dfs.withColumnRenamed("node1", "node3").withColumnRenamed("node2", "node4")
    node_xy = node_join_xy.join(node_couple_xy,
                                ((node_join_xy.node2 == node_couple_xy.node3)
                                 & (node_join_xy.node1 == node_couple_xy.node4)) |
                                ((node_join_xy.node1 == node_couple_xy.node3)
                                 & (node_join_xy.node2 == node_couple_xy.node4)) &
                                (node_join_xy.l1 == node_couple_xy.phase)
                                , 'left')
    node_xy = node_xy.withColumn('u', (node_xy['u1'] + node_xy['u2']
                                       + node_xy['ir1'] * node_xy['r1'] + node_xy['ix1'] * node_xy['x1']
                                       + node_xy['ir2'] * node_xy['r2'] + node_xy['ix2'] * node_xy['x2']) / 2)
    node_xy = node_xy.withColumn('ir', (node_xy['ir1'] + node_xy['ir2']))
    node_xy = node_xy.withColumn('ix', (node_xy['ix1'] + node_xy['ix2']))
    node_xy = node_xy.withColumn('node', lit(PARENT_NODE_BASE + q))
    node_xy = node_xy.withColumn('lv', lit(q + 2))
    node_xy = node_xy.select('DATA_TIME', 'L1', 'U')
    df_parent = node_xy.toPandas()

    # 用df_parent替换pairRDD中的父节点
    pairRDD = pairRDD.map(
        lambda r: (r[0], (df_parent if r[0] == idx_node_x else r[1]))
        )

    # 计算父节点和所有旧节点的距离，更新corrs矩阵
    # create df_parent
    inactive_set.add(idx_node_y)
    print('inactive_set', inactive_set)

    def calc_dist(item):
        r, index = item
        # df_existing = pd.DataFrame(r[1], columns=header)
        other_node, df_existing = r
        # 跳过自己和已合并节点
        if index == idx_node_x or index in inactive_set:
            return -np.inf,
        df_join = df_existing.merge(df_parent, on=('DATA_TIME', 'L1'), suffixes=('_l', '_r'))
        # TODO: coalesce?
        try:
            corr = df_join.loc[:,('U_l', 'U_r')].corr().iloc[0, 1]
        except IndexError:
            print('error')
            corr = -np.inf
        return corr,
    x_distance = np.array([o[0] for o in pairRDD.zipWithIndex().map(calc_dist).collect()])
    print('new distance')
    print(x_distance)
    print(corrs)
    corrs[idx_node_x,   idx_node_x+1:] = x_distance[idx_node_x+1:]
    corrs[:idx_node_x,idx_node_x] = x_distance[:idx_node_x]
    corrs[idx_node_y,:] = -np.inf
    corrs[:,idx_node_y] = -np.inf
    print(corrs)
    return pairRDD


def main():
    # 读入数据
    input_lh_tp_node_ui = get_origin_data()
    distinct_nodes = input_lh_tp_node_ui.select("node").distinct().sort(col("node"))
    node_names = [o[0] for o in distinct_nodes.collect()]
    print('node_names', node_names)
    global NODE_COUNT
    NODE_COUNT = len(node_names)

    # 准备主附表
    primary_list = []
    # TODO: 创建空spark df，不要用随机数
    df = pd.DataFrame(np.random.random((1, 7)))
    secondary_df = spark.createDataFrame(df, schema=['node', 'phase', 'rsquare', 'r', 'x', 'b0', 'b1'])

    # 计算距离
    a = time.time()
    pairRDD, corrs = get_corr_matrix(input_lh_tp_node_ui, node_names)
    print('距离矩阵生成时间: %s Seconds' % (time.time() - a))

    # 循环，直到仅剩两个点为止
    inactive_set = set()
    for q in range(NODE_COUNT - 2):
        a = time.time()
        print("{} correlation data num of partition : {}".format(q, input_lh_tp_node_ui.rdd.getNumPartitions()))

        # 获取最相关的两个点
        idx_node_x, idx_node_y = get_corr_max_two_node_name(corrs)
        print('idx_node_x, idx_node_y', idx_node_x, idx_node_y)
        print('correlation running time: %s Seconds' % (time.time() - a))

        # 合并节点、更新距离
            # 计算回归系数
            # idx_node_x < idx_node_y
        node_x, node_y = (node_names[i] for i in (idx_node_x, idx_node_y))
        print('node_x, node_y', node_x, node_y)

        # TODO: sort?
        param_dfs, node_join_xy = get_linear_regression_param_list(input_lh_tp_node_ui, node_x, node_y)
        input_lh_tp_node_ui = updata_node_couple(input_lh_tp_node_ui, node_join_xy, param_dfs, node_x, node_y, q)
        print('regression running time: %s Seconds' % (time.time() - a))

            # 计算父节点参数
        s1_dfs, s2_dfs, p1_list, p2_list = get_primary_secondary_single_data(param_dfs, node_x, node_y, q)
        # primary_list.append(p1_list)
        # primary_list.append(p2_list)
        secondary_df = secondary_df.union(s1_dfs)
        secondary_df = secondary_df.union(s2_dfs)

            # 更新node_names和距离矩阵
        node_names[idx_node_x] = q + PARENT_NODE_BASE
        node_names[idx_node_y] = 'ERROR'
        pairRDD = update_distance(pairRDD, node_join_xy, param_dfs, idx_node_x, idx_node_y, corrs, q, inactive_set)
        print('other running time: %s Seconds' % (time.time() - a))

            # 更新主附表
        s1_dfs, s2_dfs, p1_list, p2_list = get_primary_secondary_single_data(param_dfs, node_x, node_y, q)
        primary_list.append(p1_list)
        primary_list.append(p2_list)
        secondary_df = secondary_df.union(s1_dfs)
        secondary_df = secondary_df.union(s2_dfs)
        # secondary_df = secondary_df.coalesce(10)
        print('%s all running time: %s' % (q, time.time() - a))


    # 储存主附表
    # 主副表数据存储
    primary_table, secondary_table = save_table_data(primary_list, secondary_df)


if __name__ == '__main__':
    main()
