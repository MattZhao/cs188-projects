# bayesNet.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
from collections import defaultdict
import random
from copy import deepcopy, copy

class BayesNet(object):

    def __init__(self, variables, inputInEdges, inputOutEdges, inputVariableDomains):
        """
        Bare bones constructor for bayes nets.
        Use constructEmptyBayesNet for a nicer interface.

        variables:       An iterable of all of the variables.
        inEdges:         A dict that maps variable to otherVariable
                         when there is an edge from otherVariable to
                         variable 
        outEdges:        A dict that maps variable to otherVariable
                         where there is an edge from variable to
                         otherVariable  
        variableDomains: A dict mapping each variable to its domain (list like).

        Constructs a bayes net with edges given by inEdges and
        outEdges for each variable.
        Doesn't initialize the conditional probability table for any variables.
        """
        # Each variable is unique (so that they can be keys in dicts)
        self.__variablesSet = set(variables)
        self.__variables = sorted(list(variables))
        # self.__inEdges[v] = [u if the edge (u, v) exists]
        self.__inEdges  = inputInEdges 
        # self.__outEdges[u] = [v if the edge (u, v) exists]
        self.__outEdges = inputOutEdges

        # make sure that the edge maps contain all variables
        for variable in self.__variablesSet:
            if variable not in self.__inEdges:
                self.__inEdges[variable]  = set()
            if variable not in self.__outEdges:
                self.__outEdges[variable] = set()

        self.__variableDomainsDict = inputVariableDomains
        self.__CPTDict = {}

    def variablesSet(self):
        " Returns a copy of the set of variables in the bayes net "
        return copy(self.__variablesSet)

    def variableDomainsDict(self):
        " Returns a copy of the variable domains in the bayes net "
        return deepcopy(self.__variableDomainsDict)

    def inEdges(self):
        " Returns a copy of the incoming edges in the bayes net "
        return deepcopy(self.__inEdges)

    def outEdges(self):
        " Returns a copy of the outgoing edges in the bayes net "
        return deepcopy(self.__outEdges)

    def __str__(self):
        """
        Human-readable representation of a bayes net. 
        Prints each variable, each edge, and then each conditional probability table.
        """
        netString = "Variables: " + ", ".join([str(var) for var in self.__variablesSet]) + "\n" + \
                    "Edges: " + ", ".join([str(fromVar) + " -> " + str(toVar) \
                    for toVar in self.__variablesSet \
                    for fromVar in self.__inEdges[toVar]])
        try:
            factorsString = "Conditional Probability Tables:\n\n" + \
                            "\n ======================= \n\n".join([str(factor) for factor in self.getAllCPTsWithEvidence()])
            return netString + '\n\n' + factorsString
        except KeyError:
            return netString + '\n' + repr(self.variableDomainsDict())

    def sameGraph(self, other):
        sameVars = self.__variables == other.__variables
        sameInEdges = sorted(self.__inEdges) == sorted(other.__inEdges)
        sameOutEdges = sorted(self.__outEdges) == sorted(other.__outEdges)
        return sameVars and sameInEdges and sameOutEdges

    def linearizeVariables(self):
        """
        Returns a list of the variables in the bayes net, but in a
        linearized order (parents of a variable in the graph always
        precede it in the order).

        Useful for sampling.
        """
        inEdgesIncremental = dict([(var, edgeSet.copy()) for (var, edgeSet) in self.__inEdges.items()])
        noIncomingList = [var for var in self.__variables if len(self.__inEdges[var]) == 0]

        linearizedList = []

        while len(noIncomingList) > 0:
            currentVar = noIncomingList.pop()
            linearizedList.append(currentVar)
            for outGoingVariable in self.__outEdges[currentVar]:
                inEdgesIncremental[outGoingVariable].remove(currentVar)
                if len(inEdgesIncremental[outGoingVariable]) == 0:
                    noIncomingList.append(outGoingVariable)

        hasEdgesLeftOver = any([len(inEdgesIncremental[var]) > 0 for var in self.__variables])
        if hasEdgesLeftOver:
            raise ValueError, ("Graph has at least one cycle (not a bayes net) " + \
                               str(inEdgesIncremental))
        else:
            return linearizedList

    def getCPT(self, variable):
        """
        Returns a copy of the conditional probability table in the bayes net
        for variable.  This is instantiated as a factor.
        """
        if variable not in self.__variablesSet:
            raise ValueError, ("Variable not in bayes net: " + str(variable))
        else:
            return deepcopy(self.__CPTDict[variable])

    def setCPT(self, variable, CPT):
        """
        Sets the conditional probability table in the bayes net for
        variable.

        CPT is a Factor of the conditional probability table with variable 
        as the only unconditioned variable, and each conditioned variable
        must have an edge going into variable.
        """
        if variable not in self.__variablesSet:
            raise ValueError, ("Variable not in bayes net: " + str(variable))
        else:
            unconditionedVariables = CPT.unconditionedVariables()
            conditionedVariables = CPT.conditionedVariables()

            if len(unconditionedVariables) != 1:
                raise ValueError, ("Unconditioned variables must contain a single element for an entry" + \
                                   " in the conditional probability tables for this Bayes net\n" + \
                                  str(unconditionedVariables))


            unconditionedVariable = list(unconditionedVariables)[0]
            
            if unconditionedVariable != variable:
                raise ValueError, ("Variable in the input and the " 
                                  + "unconditionedVariable in the factor must \nagree. " +
                                  "Input variable: " + str(variable) + \
                                  " unconditioned variable: " + str(unconditionedVariable))

            for var in conditionedVariables:
                if var not in self.__inEdges[unconditionedVariable]:
                    raise ValueError, ("Conditioned variables must be all have an edge " +
                                       "going into \n the unconditionedVariable. \n" +
                                       "conditionedVariables: " + str(conditionedVariables) + \
                                       "\nparent: " + str(var))

            self.__CPTDict[variable] = deepcopy(CPT)

    def getReducedVariableDomains(self, evidenceDict):
        """
        evidenceDict: A dict with an assignment for each
                                evidence variable.

        Returns a new variableDomainsDict where each evidence
        variable's domain is the single value that it is being
        assigned to (and is otherwise unchanged).
        """
        reducedVariableDomainsDict = self.variableDomainsDict()
        for (evidenceVariable, value) in evidenceDict.items():
            reducedVariableDomainsDict[evidenceVariable] = [value]
        return reducedVariableDomainsDict

    def getCPTWithEvidence(self, variable, evidenceDict=None):
        """
        Gets a conditional probability table for a variable, where the
        assignments in evidenceDict have been performed, so
        the CPT may have less rows than what
        would be returned from getCPT.

        Input evidenceDict is optional.
        If it is not provided, the CPTs for all variables without 
        specializing the domains is provided.
        """
        if evidenceDict is None or len(evidenceDict.items()) == 0:
            return self.getCPT(variable)
        else:
            reducedVariableDomains = self.getReducedVariableDomains(evidenceDict)
            variableCPT = self.getCPT(variable)
            return variableCPT.specializeVariableDomains(reducedVariableDomains)

    def getAllCPTsWithEvidence(self, evidenceDict=None):
        """
        Returns a list of conditional probability tables (taking into
        account evidence) for all variables in the bayes net.

        Input evidenceDict is optional.
        If it is not provided, the CPTs for all variables without 
        specializing the domains is provided.
        """
        return [self.getCPTWithEvidence(var, evidenceDict) for var in self.__variablesSet]

    def easierToParseString(self, printVariableDomainsDict=False):
        " Used internally for computer-readable printing "
        returnStrings = []
        for CPT in self.getAllCPTsWithEvidence():
            # CPT has only one unconditioned variable, extract it and use as a prefix
            prefix = next(iter(CPT.unconditionedVariables()))
            returnStrings.append(CPT.easierToParseString(prefix=prefix, printVariableDomainsDict=printVariableDomainsDict))
            printVariableDomainsDict = False
        return "\n".join(returnStrings)


class Factor(object):

    def __init__(self, inputUnconditionedVariables, inputConditionedVariables, inputVariableDomainsDict):
        """
        Constructor for factors.

        Takes in as input an iterable unconditionedVariables, an iterable 
        conditionedVariables, and a variableDomainsDict as a mapping from 
        variables to domains.

        inputUnconditionedVariables is an iterable of variables (represented as strings)
            that contains the variables that are unconditioned in this factor 
        inputConditionedVariables is an iterable of variables (represented as strings)
            that contains the variables that are conditioned in this factor 
        inputVariableDomainsDict is a dictionary from variables to domains of those 
            variables (typically represented as a list but could be any iterable)

        Initializes the probability entries of all possible assignmentDicts to be 0.0
        """
        # if only one variable is passed in (not in a list), wrap it in a list
        if isinstance(inputUnconditionedVariables, str):
            inputUnconditionedVariables = [inputUnconditionedVariables]

        if isinstance(inputConditionedVariables, str):
            inputConditionedVariables = [inputConditionedVariables]

        repeatedVariables = set(inputUnconditionedVariables) & set(inputConditionedVariables)
        if repeatedVariables:
            raise ValueError, ("unconditionedVariables and conditionedVariables "\
                               "can't have repeated \n variables. Repeats:\n" +  str(repeatedVariables))


        self.__variables = tuple(inputUnconditionedVariables) + tuple(inputConditionedVariables) # variables are unique string identifiersk
        self.__variablesSet = set(self.__variables)

        if not self.__variablesSet.issubset(set(inputVariableDomainsDict.keys())): # it's okay for variableDomainsDict to have more items than needed
            raise ValueError, ("variableDomainsDict doesn't have all the input variables \n" \
                               + str(self.__variablesSet))

        self.__unconditionedVariables = set(inputUnconditionedVariables)
        self.__conditionedVariables = set(inputConditionedVariables)
        self.__variableDomainsDict = deepcopy(inputVariableDomainsDict) # dict that maps {variable : variableDomain}

        self.__variableOrders = dict([(variable, i) for i, variable in enumerate(self.__variables)]) # internal order of the variables
        self.__probDict = {} # probability values are stored in an {assignmentValuesTuple : probability} dict,
                            # since we can't index using assignmentDicts. this is why we have to sort
        products = list(itertools.product(*[inputVariableDomainsDict[variable] for variable in self.__variables]))
        for assignmentsInOrder in products:
            self.__probDict[tuple(assignmentsInOrder)] = 0.0

    def variableDomainsDict(self):
        " Retuns a copy of the variable domains in the factor "
        return deepcopy(self.__variableDomainsDict)

    def variables(self):
        " Retuns a copy of the tuple of variables in the factor "
        return copy(self.__variables)

    def variablesSet(self):
        " Retuns a copy of the set of variables in the factor "
        return copy(self.__variablesSet)

    def unconditionedVariables(self):
        " Retuns a copy of the unconditioned variables in the factor "
        return copy(self.__unconditionedVariables)

    def conditionedVariables(self):
        " Retuns a copy of the conditioned variables in the factor "
        return copy(self.__conditionedVariables)

    def __eq__(self, other):
        """
        Tests equality of two factors.

        Makes sure the unconditionedVariables,
        conditionedVariables of the two factors are the same.
        Then makes sure each table in the first is the same 
        (up to some tolerance) as the ones in the second and vice versa.
        Returns true if they are the same.
        """
        variablesEqual = (self.variablesSet() == other.variablesSet()) \
                and (set(self.unconditionedVariables()) == set(other.unconditionedVariables())) \
                and (set(self.conditionedVariables()) == set(other.conditionedVariables()))

        if not variablesEqual:
            return False

        for assignmentDict in self.getAllPossibleAssignmentDicts():
            selfProb = self.getProbability(assignmentDict)
            try:
                otherProb = other.getProbability(assignmentDict)
            except ValueError:
                return False # have different variable domains
            if abs(selfProb - otherProb) > 10e-13:
                return False

        for assignmentDict in other.getAllPossibleAssignmentDicts():
            otherProb = other.getProbability(assignmentDict)
            try:
                selfProb = self.getProbability(assignmentDict)
            except ValueError:
                return False # have different variable domains
            if abs(selfProb - otherProb) > 10e-13:
                return False
        return True

    def __ne__(self, other):
        " Tests if two factors are not equal "
        return not self.__eq__(other)

    def getProbability(self, assignmentDict):
        """ 
        Retrieval function for probability values in the factor.

        Input assignmentDict should be of the form {variable : variableValue} for all
        variables in the factor.

        assignmentDict can have more variables than the factor contains 
        (for instance, it could have an assignment for all the 
        variables in a bayes net), and it will select the right row 
        from this factor, ignoring the variables not contained within. 

        Returns the probability entry stored in the factor for that 
        combination of variable assignments.
        """
        assignmentsInOrder = self.__getAssignmentsInOrder(assignmentDict)
        if assignmentsInOrder not in self.__probDict:
            raise ValueError, ("The input assignmentDict is not contained in this factor: \n" \
                                +  str(self) + str(assignmentDict))
        else:
            return self.__probDict[assignmentsInOrder]

    def setProbability(self, assignmentDict, probability):
        """ 
        Setting function for probability values in the factor.

        Input assignmentDict should be of the form {variable : variableValue} 
        for all variables in the factor.
        assignmentDict can have more variables than the factor contains 
        (for instance, it could have an assignment for all the variables in a bayes net),
        and it will select the right row from this factor, ignoring the variables 
        not contained within. 

        Input probability is the probability that will be set within the table.
        It must be non-negative.

        Returns None
        """
        if probability < 0: 
            raise ValueError, ("Probabilty entries can't be set to negative values: " + \
                               str(probability))
        else:

            assignmentsInOrder = self.__getAssignmentsInOrder(assignmentDict)
            if assignmentsInOrder not in self.__probDict:
                raise ValueError, ("The input assignmentDict is not contained in this factor: \n" \
                                  +  str(self) + str(assignmentDict))
            else:
                self.__probDict[assignmentsInOrder] = probability

    def __getAssignmentsInOrder(self, assignmentDict):
        """
        Internal utility function for interacting with the stored
        probability dictionary.

        We would like to store a probability value for each
        assignmentDict, but dicts aren't hashable since they're
        mutable, so we can't have a dict with dicts as keys.  
        So we store the probability table in a dict where the keys are
        tuples of variable values, without the variable name
        associated with the value.

        This function takes an assignmentDict and processes it into an
        ordered tuple of values where the values are the assignments
        in assignmentDict.
        We can then use this tuple to directly index into the
        probability table dict.

        Use factor.getProbability and factor.setProbability instead,
        for a better interface.
        """
        reducedAssignmentDict = dict([(var, val) for (var, val) \
                                      in assignmentDict.items() if var in self.__variablesSet])
        variablesAndAssignments = reducedAssignmentDict.items()
        variablesAndAssignments = sorted(variablesAndAssignments, \
                                         key=lambda (var, val) : self.__variableOrders[var])
        return tuple([val for (var, val) in variablesAndAssignments])

    def getAllPossibleAssignmentDicts(self):
        """
        Use this function to get the assignmentDict for each 
        possible assignment for the combination of variables contained
        in the factor.

        Returns a list of all assignmentDicts that the factor contains
        rows for, allowing you to iterate through each row in the
        factor when combined with getProbability and setProbability).
        """
        cartesianProductOfAssignments = itertools.product(*[self.__variableDomainsDict[variable] for variable in reversed(self.__variables)])
        return [dict(zip(reversed(self.__variables), product)) for product in cartesianProductOfAssignments]


    def __str__(self):
        """
        Print a human-readable tabular representation of a factor.
        """
        printSizeDict = {}
        for variable in self.__variablesSet:
            maxPrintSize = max(len(variable), max([len(str(variableValue)) for variableValue in self.__variableDomainsDict[variable]]))
            printSizeDict[variable] = maxPrintSize

        returnString = ""

        # header with involved variables and unconditioned or unconditioned
        returnString += "P("
        returnString += ", ".join([str(unconditionedVariable) for unconditionedVariable in self.__unconditionedVariables])

        if len(self.__conditionedVariables) > 0:
            returnString += " | "
            returnString += ", ".join([str(conditionedVariable) for conditionedVariable in self.__conditionedVariables])

        returnString += ")\n\n"

        # first line of table with variable names
        varLine = " | " + " | ".join([str(unconditionedVariable)[:printSizeDict[unconditionedVariable]].center(printSizeDict[unconditionedVariable], ' ') 
                                      for unconditionedVariable in self.__unconditionedVariables])
        if len(self.__conditionedVariables) > 0:
            varLine += " | " + " | ".join([str(conditionedVariable)[:printSizeDict[conditionedVariable]].center(printSizeDict[conditionedVariable], ' ') 
                                                for conditionedVariable in self.__conditionedVariables])
        varLine += " | " + "Prob:".center(7, " ") + " |"

        varLineLength = len(varLine)

        returnString += varLine + "\n"


        # code for checking whether or not to print horizontal line
        previousConditionedAssignments = []
        if len(self.__conditionedVariables) == 0:
            returnString += " " + "".join(["-" for _ in range(varLineLength - 1)]) + "\n"

        # print out each row of table
        for assignmentDict in self.getAllPossibleAssignmentDicts():
            # variable assignments
            if len(self.__conditionedVariables) > 0:
                conditionedAssignments = [assignmentDict[conditionedVariable] for conditionedVariable in self.__conditionedVariables]
                if conditionedAssignments != previousConditionedAssignments:
                    returnString += " " + "".join(["-" for _ in range(varLineLength - 1)]) + "\n"
                previousConditionedAssignments = conditionedAssignments
            probability = self.getProbability(assignmentDict)
            returnString += " | " + " | ".join([str(assignmentDict[unconditionedVariable])[:printSizeDict[unconditionedVariable]].center(printSizeDict[unconditionedVariable], ' ')
                                                for unconditionedVariable in self.__unconditionedVariables])
            if len(self.__conditionedVariables) > 0:
                returnString += " | " + " | ".join([str(assignmentDict[conditionedVariable])[:printSizeDict[conditionedVariable]].center(printSizeDict[conditionedVariable], ' ')
                                                    for conditionedVariable in self.__conditionedVariables])

            # formatting for printing probability
            if probability is None:
                formattedProb = 'None'.center(7, ' ')
            else:
                digits = len(str(round(probability)))
                formattedProb = "%.1e" % probability if probability < 10e-2 else ("%." + str(8 - digits) + "f") % probability
            returnString += " | " + formattedProb
            returnString += " |\n"
        return returnString

    def __repr__(self):
        returnRepr = "Factor("
        initArgs = [self.__unconditionedVariables, self.__conditionedVariables, self.__variableDomainsDict]
        returnRepr += ", ".join([repr(arg) for arg in initArgs])
        returnRepr += ")"
        return returnRepr

    def easierToParseString(self, prefix=None, printVariableDomainsDict=True):
        """
        Print a representation of the bayes net that we have a parser for (in bayesNetTestClasses).
        """
        if prefix is None:
            prefix = ''
        returnString = ""
        if printVariableDomainsDict:
            returnString += 'variableDomainsDict: """\n'
            for (key, domain) in self.__variableDomainsDict.items():
                returnString += str(key) + ' : ' + " ".join([value for value in domain]) + '\n'
            returnString += '"""\n\n'

        returnString += prefix + 'unconditionedVariables: "'
        returnString += " ".join([unconditionedVariable for unconditionedVariable in self.__unconditionedVariables])
        returnString += '"\n\n'

        returnString += prefix + 'conditionedVariables: "'
        returnString += " ".join([conditionedVariable for conditionedVariable in self.__conditionedVariables])
        returnString += '"\n\n'

        returnString += prefix + 'FactorTable: """\n'
        for assignmentDict in self.getAllPossibleAssignmentDicts():
            probability = self.getProbability(assignmentDict)
            returnString += ", ".join([variable + " : " + str(assignmentDict[variable]) \
                                       for variable in self.__variables])
            returnString += " = " + str(probability) + "\n"
        returnString += '"""\n\n'
        return returnString

    def specializeVariableDomains(self, newVariableDomainsDict):
        """
        Returns a factor with the same variables as this factor
        but with the reduced variable domains given by
        newVariableDomainsDict.

        The entries in the probability are taken from the
        corresponding entries in this factor.
        """

        # Make sure that newVariableDomainsDict has smaller or equal
        # domain to factor.variableDomainsDict for all variables that
        # this factor contains.    
        oldVariableDomains = self.variableDomainsDict()
        for (variable, domain) in newVariableDomainsDict.items():
            if variable in self.variablesSet():
                oldVariableDomain = oldVariableDomains[variable]
                for value in domain:
                    if value not in oldVariableDomain:
                        raise ValueError, ("newVariableDomainsDict is not a subset of factor.variableDomainsDict ",
                                            "for variables contained in factor. " + "factor: " +  str(self) + 
                                            " newVariableDomainsDict: " + str(newVariableDomainsDict) +
                                            " factor.variableDomainsDict: " + str(self.variableDomainsDict()) +
                                            " variable: " + str(variable) +
                                            " value: " + str(value))

        newFactor = Factor(self.unconditionedVariables(), self.conditionedVariables(), newVariableDomainsDict)

        for assignmentDict in newFactor.getAllPossibleAssignmentDicts():
            newFactor.setProbability(assignmentDict, self.getProbability(assignmentDict))

        return newFactor


### bayes net construction utils

def constructEmptyBayesNet(variableList, edgeTuplesList, variableDomainsDict):
    " More convenient constructor for Bayes nets "
    variablesSet = set(variableList)
    inEdges  = defaultdict(set)   
    outEdges = defaultdict(set)   
    for (parent, child) in edgeTuplesList:
        # add the variables to the variables set
        inEdges[child].add(parent)
        outEdges[parent].add(child)

    newBayesNet = BayesNet(variablesSet, inEdges, outEdges, variableDomainsDict)
    return newBayesNet

def constructEmptyBayesNetFromString(bayesNetString):
    variables = bayesNetString.split('\n')[0][len('Variables: '):].split(', ')
    edgeStrings = bayesNetString.split('\n')[1][len('Edges: '):].split(', ')
    edgeList = [(u, v) for u, _, v in map(tuple, map(str.split, edgeStrings))]
    variableDomainsDict = eval(bayesNetString.split('\n')[2])
    return constructEmptyBayesNet(variables, edgeList, variableDomainsDict)

def constructRandomlyFilledBayesNet(variableList, edgeTuplesList, variableDomainsDict):
    " Random Bayes net constructor "
    bayesNet = constructEmptyBayesNet(variableList, edgeTuplesList, variableDomainsDict)
    fillTablesRandomly(bayesNet)
    return bayesNet


def fillTablesRandomly(bayesNet):
    " Fills a Bayes net with random variables "
    for variable in bayesNet.variablesSet():
        conditionedVariables = bayesNet.inEdges()[variable]
        conditionedVariablesList = list(conditionedVariables)
        CPT = constructAndFillFactorRandomly([variable], conditionedVariablesList, bayesNet.variableDomainsDict())
        bayesNet.setCPT(variable, CPT)

def fillWithOneConditionedAssignmentRandomly(factor, unconditionedVariables, conditionedVariables, product, variableDomainsDict):
    """ 
    Fills one subtable of a factor (given one conditional assignment).
    Makes this subtable sum to 1.
    """
    cartesianProductOfUnConditionalAssignments = itertools.product(*[variableDomainsDict[unconditionedVariable] 
                                                                   for unconditionedVariable in unconditionedVariables])
    randomFills = [max(0.0, random.uniform(-0.4, 0.8)) for variableValue in cartesianProductOfUnConditionalAssignments]
    conditionalProbabilitySum = sum(randomFills)

    # needs to sum to 1
    if abs(conditionalProbabilitySum) < 10e-13:
        randomFills[0] = 1.0
        conditionalProbabilitySum = sum(randomFills)

    cartesianProductOfUnConditionalAssignments = itertools.product(*[variableDomainsDict[unconditionedVariable] 
                                                                   for unconditionedVariable in unconditionedVariables])

    for (randomFill, variableValue) in zip(randomFills, cartesianProductOfUnConditionalAssignments):
        assignmentDict = dict(zip(list(unconditionedVariables) + list(conditionedVariables), list(variableValue) + list(product)))
        factor.setProbability(assignmentDict, randomFill / conditionalProbabilitySum)


def constructAndFillFactorRandomly(unconditionedVariables, conditionedVariables, variableDomainsDict):
    " Wrapper around Factor constructor that fills the table randomly "
    newFactor = Factor(unconditionedVariables, conditionedVariables, variableDomainsDict)
    if len(conditionedVariables) > 0:
        cartesianProductOfConditionalAssignments = itertools.product(*[variableDomainsDict[conditionedVariable] for conditionedVariable in conditionedVariables])
        for product in cartesianProductOfConditionalAssignments:
            fillWithOneConditionedAssignmentRandomly(newFactor, unconditionedVariables, conditionedVariables, product, variableDomainsDict)
    else:
        fillWithOneConditionedAssignmentRandomly(newFactor, unconditionedVariables, [], [], variableDomainsDict)
    return newFactor

def reduceBayesNetVariablesWithEvidence(bayesNet, variablesToRemove,
                                        evidenceDict):
    """
    Prunes the variables in variablesToRemove away from the Bayes net 
    and returns a new Bayes net without variablesToRemove
    """
    variablesToRemoveSet = set(variablesToRemove)
    evidenceVariables = set(evidenceDict.keys())
    if len(variablesToRemoveSet & evidenceVariables) > 0:
        raise ValueError, ("Evidence variables are in the list of variable to " + \
                           "be removed from the Bayes' net.  This is " + \
                           "undefined. Evidence: " + str(evidenceDict) + \
                           ". Variables to remove: " + str(variablesToRemoveSet))

    newVariables = bayesNet.variablesSet() - variablesToRemoveSet
    oldOutEdges = bayesNet.outEdges()
    oldInEdges  = bayesNet.inEdges()
    newOutEdges = dict()
    newInEdges  = dict()
    for variable in newVariables:
        newOutEdges[variable] = set([y for y in oldOutEdges[variable] if y in newVariables])
        newInEdges[variable]  = set([y for y in oldInEdges[variable]  if y in newVariables])

    newVariableDomainsDict = bayesNet.getReducedVariableDomains(evidenceDict)

    newBayesNet = BayesNet(newVariables, newInEdges, newOutEdges,
                           newVariableDomainsDict)

    unconditionedVariables = newVariables - evidenceVariables
    for variable in bayesNet.variablesSet():
        if variable in newVariables:
            oldCPT = bayesNet.getCPT(variable)
            evidenceVariablesParents = []
            removedVariablesParents = []
            unconditionedVariablesParents = []
            for parentVariable in oldCPT.conditionedVariables():
                if parentVariable in variablesToRemoveSet:
                    removedVariablesParents.append(parentVariable)
                elif parentVariable in evidenceVariables:
                    evidenceVariablesParents.append(parentVariable)
                else:
                    unconditionedVariablesParents.append(parentVariable)

            if variable in evidenceVariables and \
                    len(unconditionedVariablesParents) == 0:
                newCPT = Factor([variable], evidenceVariablesParents, \
                                newVariableDomainsDict)
                # only one entry in this CPT since all parents are 
                # removed or evidence variables (and thus have one entry)
                newCPT.setProbability(evidenceDict, 1.0)

            else:
                if len(removedVariablesParents) == 0:
                    newCPT = oldCPT.specializeVariableDomains(newVariableDomainsDict)
                else:
                    raise ValueError, ("Variable: " + str(variable) + \
                                       "'s parent: " + str(parentVariable) + \
                                       " is not in the reduced bayes net, " + \
                                       "so we can't unambiguously reduce the " + \
                                       "Bayes' net.")

            newBayesNet.setCPT(variable, newCPT)
        else:
            oldCPT = bayesNet.getCPT(variable)
            #for parentVariable in oldCPT.conditionedVariables():
                #if parentVariable in unconditionedVariables:
                    #raise ValueError, ("Variable " + str(variable) + \
                                       #" is to be removed but its parent " \
                                        #+ str(parentVariable) + \
                                        #" is an unconditioned variable in the " \
                                        #+ "reduced bayes net, " + \
                                        #"so we can't reduce the " + \
                                        #"Bayes' net.")


    return newBayesNet


def printStarterBayesNet():
    """
    Exploring Bayes net functions, printing, and creation.
    Pay close attention to how factors are created and modified.
    """

    # This is the example V structured Bayes' net from the lecture 
    # on Bayes' nets independence.
    # Constructing Bayes' nets: variables list
    variableList = ['Raining', 'Ballgame', 'Traffic']

    # Constructing Bayes' nets, edge list: (x, y) means edge from x to y
    edgeTuplesList = [('Raining', 'Traffic'), ('Ballgame', 'Traffic')]

    # Construct the domain for each variable (a list like)
    variableDomainsDict = {}
    variableDomainsDict['Raining']  = ['yes', 'no']
    variableDomainsDict['Ballgame'] = ['yes', 'no']
    variableDomainsDict['Traffic']  = ['yes', 'no']

    # None of the conditional probability tables are assigned yet in our Bayes' net
    bayesNet = constructEmptyBayesNet(variableList, edgeTuplesList, variableDomainsDict)

    # Create a factor for each CPT.  
    # The first input is the list of unconditioned variables in your factor,
    # the second input is the list of conditioned variables in your factor,
    # and the third input is the dict of domains for your variables.
    rainingCPT  = Factor(['Raining'], [], variableDomainsDict)

    print "Print a conditional probability table (henceforth known as a CPT) " + \
          "to see a pretty print of the variables in a factor and its " + \
          "probability table in your terminal. " + \
          "CPTs come initialized with 0 for each row in the table: \n"
    print rainingCPT

    # We use assignmentDicts to set and get probability entries from Factors.
    # An assignmentDict is a dict {variable : variableValue} of assignments 
    # of variables to values (where the variableValue must be in 
    # variableDomainsDict[variable]

    rainAssignmentDict = {'Raining' : 'yes'}
    rainingCPT.setProbability(rainAssignmentDict, 0.3)

    rainAssignmentDict = {'Raining' : 'no'}
    rainingCPT.setProbability(rainAssignmentDict, 0.7)
    
    print 'After setting entries: \n'

    print rainingCPT

    # The traffic factor has two conditioned variables and one unconditioned 
    # variable.  Each variable has a domain size of 2, so we have 
    # 2^3 = 8 possible assignments (and thus 8 rows in our probability table).

    trafficCPT  = Factor(['Traffic'], ['Raining', 'Ballgame'], variableDomainsDict)

    TRB = {'Traffic' : 'yes', 'Raining' : 'yes', 'Ballgame' : 'yes'}
    tRB = {'Traffic' : 'no',  'Raining' : 'yes', 'Ballgame' : 'yes'}
    TrB = {'Traffic' : 'yes', 'Raining' : 'no',  'Ballgame' : 'yes'}
    trB = {'Traffic' : 'no',  'Raining' : 'no',  'Ballgame' : 'yes'}
    TRb = {'Traffic' : 'yes', 'Raining' : 'yes', 'Ballgame' : 'no' }
    tRb = {'Traffic' : 'no',  'Raining' : 'yes', 'Ballgame' : 'no' }
    Trb = {'Traffic' : 'yes', 'Raining' : 'no',  'Ballgame' : 'no' }
    trb = {'Traffic' : 'no',  'Raining' : 'no',  'Ballgame' : 'no' }

    # For a CPT, we must have that the sum of the probability of all the 
    # unconditionedVariables for a given assignment of conditioned 
    # variables must sum to 1
    trafficCPT.setProbability(TRB, 0.95)
    trafficCPT.setProbability(tRB, 0.05)
    trafficCPT.setProbability(TrB, 0.90)
    trafficCPT.setProbability(trB, 0.10)
    trafficCPT.setProbability(TRb, 0.70)
    trafficCPT.setProbability(tRb, 0.30)
    trafficCPT.setProbability(Trb, 0.15)
    trafficCPT.setProbability(trb, 0.85)

    print "Note that in the table output of print for factors with conditioned " + \
          "variables, each region with a different assignment of conditioned " + \
          "variables is divided into a region in the table, separated from " + \
          "other conditioned assignments by a horizontal bar. " + \
          "If a factor is a CPT, each sub table of that factor will sum to 1. \n"
    print trafficCPT

    print "You can use factor.getAllPossibleAssignmentDicts() " + \
          "to iterate through all combinations of assignments:\n"
    for assignmentDict in trafficCPT.getAllPossibleAssignmentDicts():
        print assignmentDict

    # Fill in the ballGame CPT, very similar to raining

    ballgameCPT = Factor(['Ballgame'], [], variableDomainsDict)

    # Note that we can use assignmentDicts that contain assignments for 
    # more variables than a factor mentions.
    # Here, we pass in an assignmentDict that has 3 variable assignments 
    # but ballgameCPT only contains variable Ballgame
    ballgameCPT.setProbability(TRB, 0.05)

    ballgameCPT.setProbability(TRb, 0.95)

    print "\nLast CPT: \n"

    print ballgameCPT

    # Set the factors for the bayes net to be these CPTs
    bayesNet.setCPT('Raining',  rainingCPT)
    bayesNet.setCPT('Ballgame', ballgameCPT)
    bayesNet.setCPT('Traffic',  trafficCPT)

    print "Print a Bayes' net to see its variables, edges, and " + \
          "the CPT for each variable.\n"
    print bayesNet

    print "You can get a list of all CPTs from a Bayes' net, instantiated with " + \
          "evidence, with the getAllCPTsWithEvidence function. " + \
          "The evidenceDict input is an assignmentDict of " + \
          "(evidenceVariable, evidenceValue) pairs. " + \
          "Instantiation with evidence reduces the variable domains and thus " + \
          "selects a subset of entries from the probability table."

    evidenceDict = {'Raining' : 'yes'}
    for CPT in bayesNet.getAllCPTsWithEvidence(evidenceDict):
        print CPT

    print 'If it is empty or None, the full CPTs will be returned. \n'

    for CPT in bayesNet.getAllCPTsWithEvidence():
        print CPT

    print "If only one variable's CPT is desired, you can get just that particular " + \
          "CPT with the bayesNet.getCPT function. \n"

    print bayesNet.getCPT('Traffic')

    print bayesNet.easierToParseString()


if __name__ == "__main__":
    printStarterBayesNet()
