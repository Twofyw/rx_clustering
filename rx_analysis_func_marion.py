# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import time

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

def get_origin_data():
    '''
    得到原始数据
    '''
    # input_lh_tp_node_ui = sqlContext.read.format("jdbc").options(
        # url="jdbc:hive2://192.168.1.114:10000/zkfx_test?hive.resultset.use.unique.column.names=false",
        # driver="org.apache.hive.jdbc.HiveDriver", dbtable="lh_tp_node_ui", user="admin", password="admin").load()
    # input_lh_tp_node_ui = spark.read.csv("lh_tp_node_ui.csv", header=True, inferSchema=True)
    input_lh_tp_node_ui = spark.read.csv("100_generated_test_data.csv", header=True, inferSchema=True)
    input_lh_tp_node_ui = input_lh_tp_node_ui.filter('lv == 1')
    return input_lh_tp_node_ui


def get_corr_max_two_node_name(input_lh_tp_node_ui):
    '''
    得到电压相关性最大的两个节点的名称
    :param data: 原始数据
    :return: 节点名 （电表名称）
    '''
    # 生成两两节点组合的名称集合 node_couple
    node_array = np.array(input_lh_tp_node_ui.select("node").distinct().collect()).tolist()
    node_couple = []
    for i in range(len(node_array)):
        for j in range(i + 1, len(node_array)):
            for p in ['A', 'B', 'C']:
                if i != j:
                    node_couple.append([node_array[i][0], node_array[j][0], p])
    print('node_num: %s' %len(node_couple))
    # 利用电压相关系数找出相关性最大的两个点
    u_r2 = []
    for no1, no2, phase in node_couple:
        s1 = "node == %s" % no1
        node1 = input_lh_tp_node_ui.filter(s1)
        node1 = node1.select('data_time', 'u', 'l1').withColumnRenamed('u', 'u1')
        s2 = "node == %s" % no2
        node2 = input_lh_tp_node_ui.filter(s2)
        node2 = node2.select('data_time', 'u', 'l1').withColumnRenamed('u', 'u2').withColumnRenamed('l1', 'l2')
        node_join = node1.join(node2, (node1.data_time == node2.data_time) & (node1.l1 == node2.l2))
        node_join = node_join.coalesce(10)
        u_rsquare = node_join.corr('u1', 'u2')
        u_r2.append([no1, no2, phase, u_rsquare])
    name_u_r2 = ["no1", "no2", "phase", "u_rsquare"]
    u_r2 = pd.DataFrame(columns=name_u_r2, data=u_r2)
    node_x, node_y, phase_v, u_rsquare_best = u_r2[u_r2['u_rsquare'] == u_r2['u_rsquare'].max()].iloc[0,]
    print("In this circulation, node {} and node {} is best!".format(node_x, node_y))
    return node_x, node_y

def linear_regression(node_x, node_y, node_join_xy, phase='A'):
    '''
    两节点在单个相位上做线性回归分析
    :param node_x: node_x name
    :param node_y: node_y name
    :node_join_xy: node_x node_y拼接后的数据
    :param phase: 相位
    :return: 回归系数列表
    '''
    # 筛选出单相位的数据
    s3 = 'l1 == "%s"' % phase
    node_join_p = node_join_xy.filter(s3)
    node_join_p = node_join_p.drop('node1', 'node2', 'l1')
    assembler = VectorAssembler(inputCols=["u1", "ir1", "ix1", "ir2", "ix2"], outputCol="features")
    output = assembler.transform(node_join_p)
    label_features = output.select("features", "u2").toDF('features', 'label')
    lr = LinearRegression(maxIter=5, elasticNetParam=0.8)
    lrModel = lr.fit(label_features)
    trainingSummary = lrModel.summary
    param = [node_x, node_y, phase, trainingSummary.r2,
             lrModel.intercept,
             lrModel.coefficients[0],
             lrModel.coefficients[1],
             lrModel.coefficients[2],
             lrModel.coefficients[3],
             lrModel.coefficients[4]]
    return param

def get_linear_regression_param_list(data, node_x, node_y):
    '''
    两节点在A、B、C三个相位上分别做线性回归
    :param data: 原始数据
    :param node_x: node_x name
    :param node_y: node_y name
    :return: 不同相位的回归系数 ~ spark-df/ node_x,node_y合并后的数据
    '''
    # 生成做回归分析的数据
    s_x = "node == {}".format(node_x)
    s_y = "node == {}".format(node_y)
    nodex = data.filter(s_x)
    nodey = data.filter(s_y)
    nodex = nodex.withColumnRenamed('node', 'node1').withColumnRenamed('u', 'u1').withColumnRenamed('ir',
                                                                                                    'ir1').withColumnRenamed(
        'ix', 'ix1')
    nodey = nodey.withColumnRenamed('node', 'node2').withColumnRenamed('l1', 'l2').withColumnRenamed('u',
                                                                                                     'u2').withColumnRenamed(
        'ir', 'ir2').withColumnRenamed('ix', 'ix2').withColumnRenamed('data_time', 'data_time2')
    node_join_xy = nodex.join(nodey, ((nodex['data_time'] == nodey.data_time2) & (nodex['l1'] == nodey.l2)))
    node_join_xy = node_join_xy.select('node1', 'node2', 'data_time', 'l1', 'u1', 'ir1', 'ix1', 'u2', 'ir2', 'ix2')
    node_join_xy = node_join_xy.withColumn("ir2", node_join_xy["ir2"] * (-1))
    node_join_xy = node_join_xy.withColumn("ix2", node_join_xy["ix2"] * (-1))
    node_join_xy = node_join_xy.coalesce(10)
    # 获得两表~三相位~的回归系数列表
    param_list = []
    for phase in ['A', 'B', 'C']:
        param_list.append(linear_regression(node_x, node_y, node_join_xy, phase=phase))
    name = ['node1', 'node2', 'phase', 'rsquare', 'b0', 'b1', 'r1', 'x1', 'r2', 'x2']
    param_df = pd.DataFrame(columns=name, data=param_list)
    param_dfs = spark.createDataFrame(param_df)
    return param_dfs, node_join_xy

def updata_node_couple(data, node_join_xy, param_dfs, node_x, node_y, q):
    '''
    更新原始数据：从节点集合中删除两个子节点，添加父节点
    :param data: 原始数据
    :param param_dfs: 两节点的回归系数 ~ dfs
    :return: 删除两个子节点，添加父节点后的输入数据
    '''
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
    node_xy = node_xy.withColumn('node', lit(80 + q))
    node_xy = node_xy.withColumn('lv', lit(q + 2))
    node_xy = node_xy.select('lv', 'node', 'l1', 'data_time', 'u', 'ir', 'ix')
    # 更新数据
    s_filter = 'node != "{}" and node != "{}"'.format(node_x, node_y)
    data = data.filter(s_filter)
    new_data = data.union(node_xy)
    new_data = new_data.coalesce(10)
    new_data.select('node').distinct().show()
    return new_data

def get_primary_secondary_single_data(param_dfs, node_x, node_y, q):
    '''
    :param param_dfs: 两节点的回归系数
    :return: 主副表单条数据
    '''
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
    p1_list = [node_x, 80 + q, avg_rsquare, avg_r1, avg_x1, avg_z1, avg_b0, avg_b1]
    p2_list = [node_y, 80 + q, avg_rsquare, avg_r2, avg_x2, avg_z2, avg_b0, avg_b1]
    return s1_dfs, s2_dfs, p1_list, p2_list

def save_table_data(primary_list, secondary_df):
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


def main():
    begin = time.time()
    # 原始数据
    input_lh_tp_node_ui = get_origin_data()
    cnt = input_lh_tp_node_ui.select("node").distinct().count()

    # =====================设置需要存储主副表的数据格式================== #
    df = pd.DataFrame(np.random.random((1, 7)))
    secondary_df = spark.createDataFrame(df, schema=['node', 'phase', 'rsquare', 'r', 'x', 'b0', 'b1'])
    primary_list = []
    # =================================================================== #

    # 创建循环体，每次循环减少一个点直到仅剩两个点为止
    for q in range(cnt - 2):
        a = time.time()
        # 得到电压相关系数相关性最大的两个点
        print("{} correlation data num of partition : {}".format(q, input_lh_tp_node_ui.rdd.getNumPartitions()))
        node_x, node_y = get_corr_max_two_node_name(input_lh_tp_node_ui)
        print('correlation running time: %s Seconds' % (time.time() - a))

        # 得到两个点线性回归后的回归系数 - dfs
        print("{} regression data num of partition : {}".format(q, input_lh_tp_node_ui.rdd.getNumPartitions()))
        param_dfs, node_join_xy = get_linear_regression_param_list(input_lh_tp_node_ui, node_x, node_y)
        node_join_xy = node_join_xy.coalesce(10)
        print('regression running time: %s Seconds' % (time.time() - a))

        # 更新输入数据：从原始数据中删除两个子节点，添加新的父节点
        input_lh_tp_node_ui = updata_node_couple(input_lh_tp_node_ui, node_join_xy, param_dfs, node_x, node_y, q)
        print("before modify,input_lh_tp_node_ui num of partition : {}".format(input_lh_tp_node_ui.rdd.getNumPartitions()))
        input_lh_tp_node_ui = input_lh_tp_node_ui.coalesce(10)
        print("after modify,input_lh_tp_node_ui's partition: {}".format(input_lh_tp_node_ui.rdd.getNumPartitions()))
        print('%s loop new_data lines_num:%s' % (q, input_lh_tp_node_ui.count()))
        print('other running time: %s Seconds' % (time.time() - a))

        # 主副表单条数据生成、添加
        s1_dfs, s2_dfs, p1_list, p2_list = get_primary_secondary_single_data(param_dfs, node_x, node_y, q)
        primary_list.append(p1_list)
        primary_list.append(p2_list)
        secondary_df = secondary_df.union(s1_dfs)
        secondary_df = secondary_df.union(s2_dfs)
        secondary_df = secondary_df.coalesce(10)
        print('%s all running time: %s' % (q, time.time() - a))

    # 主副表数据存储
    primary_table, secondary_table = save_table_data(primary_list, secondary_df)
    print(time.time() - begin)
    return primary_table, secondary_table

if __name__ == '__main__':
    primary_table, secondary_table = main()
