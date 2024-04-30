import numpy as np
import casadi as ca
import itertools
from builders import StlTask, TimeInterval, TemporalOperator, AlwaysOperator, EventuallyOperator, PredicateFunction
from graph_module import EdgeTaskContainer
import networkx as nx
from typing import Tuple , List, Dict

globalOptimizer = ca.Opti()


def edgeSet(path:list, isCycle:bool=False) -> List[Tuple[int,int]]:
    """Given a list of nodes, it returns the edges of the path as (n,n+1) 
    Inputs 
    ------------------------------------------------------------------------------------------------
    path (list<float>) : list of nodes in the path
    
    Output 
    ------------------------------------------------------------------------------------------------
    edges (list<tuple>): list of tuples (n,n+1)
    """

    if not isCycle:
      edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    elif isCycle: # due to how networkx returns edges
      edges = [(path[i], path[i+1]) for i in range(-1, len(path)-1)]
        
    
    return edges


def pathMinkowskiSumVertices(listOfFormulas : List[StlTask], edgeList: List[Tuple[int,int]]) -> ca.MX  :
    """
    Given a set of edges with assigned formulas, it computes the vertices of the minkowski sum for the superlevel sets of the given formulas.
    Each formula is define alon one edge of the edgeList in the same sequence as the formulas are given. The edge information is used to 
    decide in which verse we should take the formulas if their direction of definiton is different from the direction of the path for example
    
    Inputs
    --------------------------------------------------
    listOfFormulas (list<STLFormula>)
    
    Outputs
    -----------------------------------------
    minkowshySumVertices (cvxpy.Variable): returns a matrix where each column is a vertex of the Minkowski sum of the hypercubes defined by each edge in the list of edges
    """
    
    if len(listOfFormulas) != len(edgeList) :
        raise ValueError("list of formulas must have the same length of edges list")
    
    stateSpaceDimension   = listOfFormulas[0].predicate.stateSpaceDimension
    cartesianProductSets  = [[-1,1],]*stateSpaceDimension
    hypercubeVertices     = np.array(list(itertools.product(*cartesianProductSets))).T # All vertices of hypercube centered at the origin (Each vertex is a column)
    
    center = 0 # center of the hypercube
    nuSum  = 0 # dimensions of the hypercube
    
    for formula,edgeTuple in zip(listOfFormulas,edgeList) :
    
        if formula.edgeTuple != edgeTuple and formula.edgeTuple!= (edgeTuple[1],edgeTuple[0]) :
            raise ValueError("edge along the path is not maching the corresponding formula. Make sure that the edges order and the formulas order is correct")
        
        elif formula.edgeTuple != edgeTuple : # case in which the directions are not matching
            if formula.predicate.isParametric :
                center = center - formula.centerVar
                nuSum  = nuSum  + formula.nuVar
            else : # case in which the formula is not parameteric 
                if formula.predicate._isApproximated :
                    center = center - formula.predicate.optimalApproximationCenter
                    nuSum  = nuSum  + formula.predicate.optimalApproximationNu
                else :
                    raise Exception("formula does not have an approximation available. PLease replace formula with its approximation before calling this method")
        else : # case in which the directions are matching
            if formula.predicate.isParametric :
                center = center + formula.centerVar
                nuSum  = nuSum  + formula.nuVar
            else : # case in which the formula is not parameteric 
                if formula.predicate._isApproximated :
                    center = center + formula.predicate.optimalApproximationCenter
                    nuSum  = nuSum  + formula.predicate.optimalApproximationNu
                else :
                    raise Exception("formula does not have an approximation available. PLease replace formula with its approximation before calling this method")
        
    minkowshySumVertices = center + hypercubeVertices*nuSum/2 # find final hypercube dimension
    
    return minkowshySumVertices


def computeNewTaskGraph(task_graph:nx.Graph, comm_graph:nx.Graph, task_edges:List[Tuple], comm_info:Dict[str, Dict], start_position: Dict[int, np.ndarray], problemDimension = 2, maxInterRobotDistance = 3)-> nx.Graph: 
    """ Solves the task decomposition completely"""
    
    numberOfVerticesHypercube = 2**problemDimension
    originalTaskGraph = task_graph.copy()
    

    pathsList                   : list[list[int]] = []
    pathConstraints             : list[ca.MX] = []
    overloadingConstraints      : list[ca.MX] = []
    cyclesConstraints           : list[ca.MX] = []
    maxCommunicationConstraints : list[ca.MX] = []
    positiveNuConstraint        : list[ca.MX] = []

    decompositionOccurred = False
    decompositionSolved = []

    for task_name, (task_key, task_dict) in enumerate(comm_info.items()):

        edge = task_dict["EDGE"]
        edge_container: EdgeTaskContainer = task_graph[edge[0]][edge[1]]["container"] 
        has_tasks = len(edge_container.task_list) > 0
        isCommunicating = task_dict["COMMUNICATE"]

        if (not isCommunicating) and (has_tasks): 
            decompositionOccurred = True
            
            # retrive all the formulas to be decomposed
            formulasToBeDecomposed: list[StlTask] = [task for task in edge_container.task_list]
            
            
            # path finding and grouping nodes
            path = nx.shortest_path(comm_graph,source=edge[0],target=edge[1]) # path of agents from start to end
            pathsList.append(path) # save sources list for later plotting
            # for each formula to be decomposed we will have n subformulas with n being the length of the path we select.

            for formula in formulasToBeDecomposed : # add a new set of formulas for each edge
                edgeSubformulas : list[StlTask] = [] # list of subformulas associate to one orginal formula. you have as many subformulas as number of edges
                
          


                originalTemporalOperator       = formula.temporal_operator                      # get time interval of the orginal operator
                originalPredicate              = formula.predicate                    # get the predicate function
                originalEdgeTuple              = tuple(formula.contributing_agents)             # get the edge tuple
                
                if originalEdgeTuple == (edge[0],edge[1]) : #case the direction is correct
                    edgesThroughPath = edgeSet(path=path) # find edges along the path
                else :
                    edgesThroughPath =  edgeSet(path=path[::-1]) # we reverse the path. This is to follow the specification direction
                
                
                for sourceNode,targetNode in edgesThroughPath:
                    
                    # create a new parameteric subformula object
                    subformula = StlTask(predicate=PredicateFunction(sourceNode=sourceNode, targetNode=targetNode), temporal_operator=originalTemporalOperator)
                    
                    # warm start of the variables involved in the optimization TODO: check if you have a better warm start base on the specification you have. Maybe some more intelligen heuristic
                    # globalOptimizer.set_initial(subformula.centerVar , MASgraph.nodes[targetNode]["pos"]-MASgraph.nodes[sourceNode]["pos"]) 
                    
                    globalOptimizer.set_initial(subformula.centerVar , start_position[sourceNode]-start_position[targetNode]) 

# Unknown: Opti decision variable 'opti0_x_1' of shape 2x1, belonging to a different instance of Opti.
# The error message "Opti decision variable 'opti0_x_1' of shape 2x1, belonging to a 
# different instance of Opti" suggests that the variable `subformula.centerVar` you're trying to set the 
# initial value for belongs to a different instance of the optimizer (`Opti`). 

                    globalOptimizer.set_initial(subformula.nuVar   , np.ones(problemDimension)*4)

                    # add subformulas to the current path
                    edgeSubformulas.append(subformula)
                    subformulaVertices = subformula.getHypercubeVertices(sourceNode=sourceNode,targetNode=targetNode)
                    
                    # set positivity of dimensions vector nu
                    positiveNuConstraint.append(-np.eye(problemDimension)@subformula.nuVar<=np.zeros((problemDimension,1))) # constraint on positivity of the dimension variable
                    task_graph[sourceNode][targetNode]["container"].add_tasks(subformula) # add the subformula to the edge container


                    # Set maximum distance constraint for each hypercube vertex
                    if maxInterRobotDistance != None :
                        for jj in range(numberOfVerticesHypercube) : 
                            maxCommunicationConstraints.append(ca.norm_2(subformulaVertices[:,jj])<=maxInterRobotDistance)     
                   
                
                # now set that the final i sum has to stay inside the original predicate
                minowkySumVertices  = pathMinkowskiSumVertices( edgeSubformulas ,  edgesThroughPath)  # return the symbolic vertices f the hypercube to define the constraints
                for jj in range(numberOfVerticesHypercube) :
                    # ======= Does not Work with my predicates ========
                    pathConstraints.append(originalPredicate.function(minowkySumVertices[:,jj])<=0) # for each vertex of the minkowski sum ensure they are inside the original predicate superlevel-set
            
            decompositionSolved.append((path,edgeSubformulas))    

            # mark all the used edges for the optimization
            edgesThroughPath = edgeSet(path=path) # find edges along the path
            # flag the edges applied for the optimization 
            for sourceNode,targetNode in   edgesThroughPath :
                    task_graph.edges[sourceNode,targetNode]["edgeObj"].flagOptimizationInvolvement()
            
            
    # # Now we check the cycle constraints on the graph as a first step and we then check the overloading constraints as a second step
    # TaskGraph       = graph.copy()
    # noTaskEdges     = [(i,j) for i,j,attr in edges if not attr["edgeObj"].hasSpecifications] 
    # noCommunication = [(i,j) for i,j,attr in edges if not attr["edgeObj"].isCommunicating] # because I have rewritten those specification
    # TaskGraph.remove_edges_from(noTaskEdges)
    # TaskGraph.remove_edges_from(noCommunication)
    
    # if decompositionOccurred :
    #     # adding cycles constraints to the optimization problem
    #     cycles :list[list[int]]   = sorted(nx.simple_cycles(TaskGraph))
    #     cycles = [cycle for cycle in cycles if len(cycle)>1] # eliminate self loopscycles)
    #     for omega in cycles :
    #         cycleEdges    = edgeSet(omega,isCycle=True)
    #         cycleEdgesObj :list[list[GraphEdge]] = [graph.edges[i,j]["edgeObj"] for i,j in cycleEdges ] 
    #         cyclesConstraints += createCycleClosureConstraint(cycleEdgeObjs=cycleEdgesObj,cycleEdges=cycleEdges)
            
    #     # now we compute the overloading constraints on a single objects
    #     # one line of overloading constraints
    #     optimisedEdges = [(i,j,edgeDict["edgeObj"]) for i,j,edgeDict in graph.edges(data=True) if edgeDict["edgeObj"].isInvolvedInOptimization]
    #     for i,j,edgeObj in optimisedEdges  :
    #         print(i,j)
    #         overloadingConstraints += computeOverloadingConstraints(edgeObj)
        
                
    #     # #########################################################################################################
    #     # # OPTIMIZATION
    #     # #########################################################################################################

    #     cost = 0 # compute cost for parameetric formulas
    #     for i,j,edgeObj in optimisedEdges :
    #         for formula in edgeObj.formulasList :
    #             if formula.isParametric :
    #                 cost = cost + 1/computeVolume(formula.nuVar)
            
            
    #     constraints = [*maxCommunicationConstraints,*positiveNuConstraint,*pathConstraints,*cyclesConstraints,*overloadingConstraints]
    #     globalOptimizer.subject_to(constraints) # Maximum Distance of a constraint


    #     globalOptimizer.solver("ipopt")
    #     solution = globalOptimizer.solve()


    #     ###########################################################################################################
    #     # PRINT SOLUTION
    #     #########################################################################################################

    #     newFormulasCount = 0
    #     for i,j,edgeObject in optimisedEdges :
    #         newFormulasCount += len([formula for formula in edgeObject.formulasList if formula.isParametric])
        
        
    #     print("-----------------------------------------")   
    #     print("Internal Report")   
    #     print("-----------------------------------------")   
    #     print(f"Total number of formulas created : {newFormulasCount}")   
    #     print("---------Found Solution------------------") 
    #     for path,formulas in decompositionSolved :   
    #         print("path: ",path)
    #         for formula in formulas:
    #             print("edge      : (" + str(formula.sourceNode)+ "," + str(formula.targetNode) + ")")
    #             print("vector    : "+ str(solution.value(formula.centerVar)))
    #             print("dimension : "+ str(solution.value(formula.nuVar)))
    #             print("formua ID : " + str(id(formula)))
    #             print("Operator  : "+ formula.temporalOperator)
    #             # turn predicates from parameteric to no parameteric
                
                
    return task_graph#TaskGraph