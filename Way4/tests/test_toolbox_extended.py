from way4.toolbox import *

def d(a,b): return abs(a-b)

def test_probability_graph_and_optimization():
    assert bayesian_update([.5,.5],[.2,.8]) == [.2,.8]
    assert dijkstra({0:[(1,2)],1:[(2,3)]},0,2)[1] == 5
    assert astar({0:[(1,2)],1:[(2,3)]},0,2,lambda _:0)[1] == 5
    assert cvar([1,10], alpha=.5) == 10

def test_gtsp_bounds_and_clustering():
    route, value = generalized_tsp([[1,2],[4,5]], 0, d)
    assert value == 4 and len(route) == 2
    assert mst_lower_bound([0,2,5], d) == 5
    assert one_tree_lower_bound([0,2,5], d) == 10
    centers, groups = kmeans([(0,0),(1,0),(10,0)], 2)
    assert len(centers) == 2 and sum(map(len, groups)) == 3
    assert len(dbscan([(0,0),(0.1,0)], .2, 2)) == 2
