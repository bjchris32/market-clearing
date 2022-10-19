# Implementation Plan of Market Clearing with Alternating BFS
# Class PerfectMatch
    # [X] initialize the payoffs
    # [X]# adjust the seller prices
        # calculate buyer payoffs given the new seller prices
        # build adjacency matix
    # [X] adjust seller prices
    # [X] record auction potential
    # [X] keep enlarging the match or find the constricted set with BFS if the matching number is less then 100
    # [ ] export the market-clearing csv file
# Class BFS
    #[X] use alternating BFS:
    #[X] trivial case: find the initial matching ---> fetching the first unmatched node using alternating BFS
    #[X] if there is a augmenting path
        # use it to enlarge the matching and continue with the new matching
            # How to enlarge the matching?
                # when searching the node in odd layer without child
                # alternate matched to unmatched edge, and
                # alternate unmatched to matched edge -> store in temp matched/unmatched array, or record the nodes in each layer
    #[X] if there is no augmenting path,
        # return the constricted set(set of nodes in even layer)
    # How to find augmenting path?
        # build odd layer(seller):
            # use non-matching edges to discover new nodes in sellers
        # build even layer(buyer):
            # case1) use matched edges to discover new nodes in buyers
            # case2) if there is a seller node without child(buyer), then we discovered an augmented path
import numpy as np
import csv

# class AlternatingBFS:
# Example with augmented set:
# adjacency_matrix = [
#     [1, 1, 0, 0],
#     [1, 0, 0, 0],
#     [0, 1, 1, 1],
#     [0, 1, 1, 0]
# ]
# # matched_buyers=[1, 2, 3, -1]
# # augmented_path = [0, 5, 2, 7]
# matched_buyers = [-1, -1, -1, -1]
# # result: self.matched_buyer = [1, 0, 3, 2]
# bfs = AlternatingBFS(adjacency_matrix, matched_buyers)
# bfs.enlarge_matching()

# Example without augmented set:
# adjacency_matrix = [
#     [1, 1, 0, 0],
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 1, 1, 1]
# ]
# matched_buyers = [-1, -1, -1, -1]
# bfs = AlternatingBFS(adjacency_matrix, matched_buyers)
# bfs.enlarge_matching()
# print("the constricted set is ==========", bfs.constricted_set())
# [0, 1, 2]
class AlternatingBFS:
    def __init__(self, adjacency_graph):
        self.adjacency_graph = adjacency_graph
        self.perfect_matching = len(self.adjacency_graph)
        self.matched_buyer = [-1] * self.perfect_matching # list of buyers, where the index is the corresponding seller

    def enlarge_matching(self):
        unmatched_buyer_index = self.unmatched_buyer_index()
        while(unmatched_buyer_index != None):
            augmented_path = self.bfs_augmented_path(unmatched_buyer_index)
            if len(augmented_path) == 0:
                break
            self.reverse_original_matching(augmented_path)
            unmatched_buyer_index = self.unmatched_buyer_index()

    def bfs_augmented_path(self, first_unmatched_buyer):
        # start traversing from unmatched_buyer with BFS
        # index 0-99 is for buyers and index 100-199 is for sellers
        self.level = [-1] * self.perfect_matching * 2
        visited = [None] * self.perfect_matching * 2

        queue = []
        queue.append(first_unmatched_buyer)
        visited[first_unmatched_buyer] = True
        self.level[first_unmatched_buyer] = 0
        parent_mapping = {}

        while queue:
            current_node = queue.pop(0)
            if self.level[current_node] % 2 == 0:
                # current_node is buyer
                index = 0
                while index < len(self.adjacency_graph[current_node]):
                    if (self.adjacency_graph[current_node][index] == 1 and
                          not visited[index+self.perfect_matching] and
                          self.buyer_unmatched_seller(index, current_node)):
                        queue.append(index+self.perfect_matching)
                        visited[index+self.perfect_matching] = True
                        self.level[index+self.perfect_matching] = self.level[current_node] + 1
                        parent_mapping[index+self.perfect_matching] = current_node
                        
                    index += 1
            else:
                # current_node is seller
                current_node = current_node - self.perfect_matching
                buyer_count = 0
                index = 0
                while index < len(self.adjacency_graph):
                    if (self.adjacency_graph[index][current_node] == 1 and
                          not visited[index] and
                          self.seller_matched_buyer(current_node, index)):
                        queue.append(index)
                        visited[index] = True
                        self.level[index] = self.level[current_node+self.perfect_matching] + 1
                        parent_mapping[index] = current_node+self.perfect_matching
                        buyer_count += 1
                    index += 1
                # early return if # augmented path found
                if buyer_count == 0:
                    return self.backtrace(parent_mapping, first_unmatched_buyer, current_node+self.perfect_matching)
        return [] # Failed to search for augmented path with alternating BFS

    def backtrace(self, parent_mapping, start, end):
        path = [end]
        if not parent_mapping:
            return path
        while path[-1] != start:
            path.append(parent_mapping[path[-1]])
        path.reverse()
        return path

    def unmatched_buyer_index(self):
        unmatch_buyers = list(set(list(range(self.perfect_matching))) - set(self.matched_buyer))
        if len(unmatch_buyers) == 0:
            return None
        else:
            return unmatch_buyers[0]

    def seller_matched_buyer(self, seller_index, buyer_index):
        if self.matched_buyer[seller_index] == buyer_index:
            return True
        else:
            return False

    def buyer_unmatched_seller(self, seller_index, buyer_index):
        # buyer has relation to the seller index according to in adjacency graph,
        # but the seller does not match to the buyer according to self.matched_buyer[seller_index]
        if self.matched_buyer[seller_index] != buyer_index and self.adjacency_graph[buyer_index][seller_index] == 1:
            return True
        else:
            return False

    # iterate every two element and modify self.matched_buyer
    # in even round match the relation
    # in odd round unmatch the relation
    def reverse_original_matching(self, augmented_path):
        # iterate every two element to negate the original matching
        index = 0
        while index < len(augmented_path):
            buyer_index = augmented_path[index]
            seller_index = augmented_path[index+1] - self.perfect_matching            
            self.matched_buyer[seller_index] = buyer_index
            index += 2

    def constricted_set(self):
        if self.unmatched_buyer_index() == None:
            return []
        else:
            constricted_set = []
            # only iterate the levels of buyer
            for i in range(self.perfect_matching):
                # only buyers with levels set will be append to constricted set
                if self.level[i] % 2 == 0:
                    constricted_set.append(i)

            return constricted_set
    
    def number_of_matched_pairs(self):
        return self.perfect_matching - self.matched_buyer.count(-1)

class PerfectMatch:
    def __init__(self, number_of_matches, file_name = './preference.csv'):
        self.number_of_matches = number_of_matches
        self.read_buyers_valuations_from_file(file_name)
        self.prices = np.zeros((self.number_of_matches))
        self.buyer_payoffs = np.zeros((self.number_of_matches, self.number_of_matches))
        self.buyer_potential = 0

    def find_matches(self):
        number_of_matches = 0
        round = 0
        while(number_of_matches < self.number_of_matches):
            constricted_buyers = []
            round += 1
            self.construct_buyer_payoff()
            adjacency_matrix = self.build_adjacency_matrix(self.buyer_payoffs)
            bfs = AlternatingBFS(adjacency_matrix)
            bfs.enlarge_matching()
            number_of_matches = bfs.number_of_matched_pairs()

            constricted_buyers = bfs.constricted_set()
            if len(constricted_buyers):
                sellers_to_adjust_prices = self.neighbors_of_constricted_buyers(adjacency_matrix, constricted_buyers)
                if len(sellers_to_adjust_prices):
                    self.adjust_seller_prices(sellers_to_adjust_prices)
            print("auction potential = ", self.auction_potential())
            # record the auction potential

        # generate the perfect match file
        print("in round = ", round)
        print("At last, self.prices === ", self.prices)

    def build_adjacency_matrix(self, buyer_payoffs):
        matrix = np.zeros((self.number_of_matches, self.number_of_matches))
        buyer_index = 0
        while(buyer_index < len(buyer_payoffs)):
            max_payoff = max(buyer_payoffs[buyer_index])
            payoff_index = 0
            while(payoff_index < len(buyer_payoffs[0])):
                if buyer_payoffs[buyer_index][payoff_index] == max_payoff:
                    matrix[buyer_index][payoff_index] = 1
                payoff_index += 1
            buyer_index += 1

        return matrix
    
    def neighbors_of_constricted_buyers(self, adjacency_matrix, constricted_buyers):
        neighbors_of_constricted_buyers = []
        for i in constricted_buyers:
            for j in range(self.number_of_matches):
                if adjacency_matrix[i][j] == 1:
                    neighbors_of_constricted_buyers.append(j)

        return list(set(neighbors_of_constricted_buyers))

    def construct_buyer_payoff(self):
        self.buyer_payoffs = np.zeros((self.number_of_matches, self.number_of_matches))
        for i in range(self.number_of_matches):
            for j in range(self.number_of_matches):
                self.buyer_payoffs[i][j] = self.buyer_valuations[i][j] - self.prices[j]
        self.calculate_buyer_potential()

    def adjust_seller_prices(self, sellers):
        for i in sellers:
            self.prices[i] += 1
        minimum_price = min(self.prices)
        # reduce all prices by minimum price
        for i in range(len(self.prices)):
            self.prices[i] -= minimum_price
        print("after adjusting the prices = ", self.prices)

    def calculate_buyer_potential(self):
        sum_of_buyer_potential = 0
        for i in range(0, self.number_of_matches):
            sum_of_buyer_potential += max(self.buyer_payoffs[i])
        self.buyer_potential = sum_of_buyer_potential
    
    def read_buyers_valuations_from_file(self, file_name):
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.buyer_valuations = np.zeros((self.number_of_matches, self.number_of_matches))
            i = 0
            for row in reader:
                j = 0
                for column in row:
                    if j == 0:
                        # do not read the first column
                        pass
                    else:
                        self.buyer_valuations[i][j - 1] = column
                    j += 1
                i += 1

    def auction_potential(self):
        return (self.prices.sum() + self.buyer_potential)

perfect_match = PerfectMatch(100)
perfect_match.find_matches()