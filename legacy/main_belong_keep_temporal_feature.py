from utility import *
from collections import defaultdict
import numpy as np
import pickle
from gurobipy import Model, GRB, quicksum

with open('../original/naval_dataset.pkl', 'rb') as f:
    data_original, true_labels, data1, true_labels1 = pickle.load(f)
    true_labels = np.where(true_labels == -1, 0, true_labels)
    true_labels1 = np.where(true_labels1 == -1, 0, true_labels1)
data_full = np.vstack((data_original, data1))
true_labels_full = np.hstack((true_labels, true_labels1))

tic()
''' data preparation '''
# data shape N by F by T, N is the number of samples, F is the number of features, T is the number of time steps
np.random.seed(42)
import numpy as np
num_sample = 100
indices = np.random.choice(len(data_full), num_sample, replace=False)
# Use the selected indices to get the samples and their corresponding labels
data = data_full[indices]
true_labels = true_labels_full[indices]
data, true_labels, index_mapping = augment_data_with_new_feature(data, true_labels) #augment_data_with_label_feature
data_count = len(data)
time_lower_limits = 0
time_upper_limits = len(data[0][0]) - 1
comb = generate_combinations(time_lower_limits, time_upper_limits, window_interval=500)
num_labels = len(np.unique(true_labels))
print('number of classes:', num_labels)
num_features = data.shape[1]-1 # subtract 1 for the label

''' Tree Construction '''
depth = 3
n, decision_nodes, leaf_nodes = generate_nodes(depth)
n_de = len(decision_nodes)
m = Model("TLI")

operations = ['less', 'greater']
discrete_operations = ['belong']
features = [f's{i + 1}' for i in range(num_features)] # features name convention s1, s2, s3, ...
discrete_features = ['s3'] # discrete features
sides = ['l', 'r']  # stand for left and right

# Create variables
op_vars = {}
for op in operations:
    for feature in features:
        op_vars[f'{op}_{feature}'] = m.addVars(n_de, vtype=GRB.BINARY, name=f"{op}than_{feature}")
# Update op_vars for discrete features
for op in discrete_operations:
    for feature in discrete_features:
        op_vars[f'{op}_{feature}'] = m.addVars(n_de, vtype=GRB.BINARY, name=f"{op}_{feature}")


w = {label: m.addVars(n, vtype=GRB.BINARY, name=f"label_{label}") for label in range(num_labels)}

for i in leaf_nodes:
    m.addConstr(sum(w[label][i] for label in range(num_labels)) == 1, name=f'leaf_label_constraint_{i}')

for i in decision_nodes:
    m.addConstr(sum(op_vars[f'{op}_{feature}'][i] for op in operations for feature in features)
                + sum(op_vars[f'{op}_{feature}'][i] for op in discrete_operations for feature in discrete_features)
                + sum(w[label][i] for label in range(num_labels)) == 1,
                name=f'decision_node_constraint_{i}')

z = {}
for i in range(0, data_count):
    for j in range(0, n):
        z[i, j] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

z_t = {}
for i in range(data_count):
    for j in range(n):
        z_t[i, j] = m.addVar(vtype=GRB.BINARY, name=f"z_t_{i}_{j}")
        m.addConstr(z_t[i, j] <= w[true_labels[i]][j], name=f'flow_constraint_{i}_{j}')

z_s = {}
for i in range(0, data_count):
    z_s[i] = m.addVar(vtype=GRB.BINARY, name=f"z_s_{i}")
    m.addConstr(z_s[i] == z[i, 0], name=f'flow_constraint_{i}')


'''
Flow constraint
'''
for i in range(0, data_count):
    for node_id in decision_nodes:  # iterate only non-leaf nodes
        left_child, right_child = 2 * node_id + 1, 2 * node_id + 2
        # Data item at parent must be at exactly one child
        m.addConstr(z[i, node_id] == z[i, left_child] + z[i, right_child] + z_t[i, node_id],
                    f"node_flow_{i}_{node_id}")

for i in range(0, data_count):
    for node_id in leaf_nodes:  # iterate leaf nodes
        m.addConstr(z[i, node_id] == z_t[i, node_id], f"leaf_flow_{i}_{node_id}")

threshold = m.addVars(n_de, vtype=GRB.CONTINUOUS, name="threshold")  # threshold for each decision node

time_window = {
    de_node_index: m.addVars(comb, vtype=GRB.BINARY, name=f"decision_{de_node_index}_var_greater") for de_node_index
    in range(n_de)}

for i in decision_nodes:
    m.addConstr(
        sum(op_vars[f'{op}_{feature}'][i] for op in operations for feature in features)
        + sum(op_vars[f'{op}_{feature}'][i] for op in discrete_operations for feature in discrete_features)
        == quicksum(time_window[i][com] for com in comb),
        name=f'time_window_constraint_{i}'
    )


rho_vars = {}
max_rho = 200
for op in operations:
    for feature in features:
        rho_vars[f'rho_{op}_{feature}'] = {}
        for i in range(data_count):
            for node in decision_nodes:
                rho_vars[f'rho_{op}_{feature}'][(i, node)] = m.addVar(
                    lb=-max_rho,
                    ub=max_rho,
                    vtype=GRB.CONTINUOUS,
                    name=f"rho_{op}_{feature}_{i}_{node}"
                )

for op in discrete_operations: # discrete rho can only be 0 or 1
    for feature in discrete_features:
        rho_vars[f'rho_{op}_{feature}'] = {}
        for i in range(data_count):
            for node in decision_nodes:
                rho_vars[f'rho_{op}_{feature}'][(i, node)] = m.addVar(
                    vtype=GRB.CONTINUOUS,
                    name=f"rho_{op}_{feature}_{i}_{node}"
                )

for i in range(data_count):
    for node in decision_nodes:
        for op, sign in [('less', 1), ('greater', -1)]:
            for feature_idx, feature in enumerate(features):
                m.addConstr(
                    quicksum(
                        time_window[node][com] * get_rho_git(com, data[i][feature_idx], int(op == 'greater'))
                        for com in comb
                    ) + sign * threshold[node] == rho_vars[f'rho_{op}_{feature}'][(i, node)],
                    name=f"rho_constraint_{op}_{feature}_{i}_{node}"
                )
        for op, sign in [('belong', 'less'), ('belong', 'greater')]:
            for feature_idx, feature in enumerate(discrete_features):
                m.addConstr(
                    quicksum(
                        time_window[node][com] * get_discrete_git(com, data[i][feature_idx+len(feature)], int(sign == 'greater'))
                        for com in comb
                    ) == rho_vars[f'rho_{op}_{feature}'][(i, node)],
                    name=f"rho_constraint_{op}_{feature}_{i}_{node}"
                )

'''Soundness Constraints'''
# Big M for indicator constraints
M = 250
epsilon = 0
# Create dictionaries to store variables
y_vars = {f"y_{op}_{feature}": {} for op in operations for feature in features}
min_vars = {f"min_{op}_{feature}": {} for op in operations for feature in features}

min_vars_discrete = {f"min_{op}_{feature}": {} for op in discrete_operations for feature in discrete_features}

for i in range(data_count):
    for j in range(n_de):
        for feature in discrete_features:
            for op in discrete_operations:
                min_vars_discrete[f"min_{op}_{feature}"][(i, j)] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"min{op}_{feature}_{i}_{j}"
                )

# Create variables
for i in range(data_count):
    for j in range(n_de):
        for op in operations:
            for feature in features:
                y_vars[f"y_{op}_{feature}"][(i, j)] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"y_{op}_{feature}_{i}_{j}"
                )
                min_vars[f"min_{op}_{feature}"][(i, j)] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"min_{op}_{feature}_{i}_{j}"
                )

# Add constraints
for i in range(data_count):
    for node in decision_nodes:
        node_id = node
        left_child, right_child = 2 * node_id + 1, 2 * node_id + 2

        for op in operations:
            for feature in features:
                # Indicator constraints
                m.addConstr(
                    rho_vars[f'rho_{op}_{feature}'][(i, node)] <= M * y_vars[f"y_{op}_{feature}"][(i, node)] + epsilon,
                    name=f"indicator_constraint_{op}_{feature}_{i}_{node}_1"
                )
                m.addConstr(
                    -rho_vars[f'rho_{op}_{feature}'][(i, node)] <= M * (
                            1 - y_vars[f"y_{op}_{feature}"][(i, node)]) + epsilon,
                    name=f"indicator_constraint_{op}_{feature}_{i}_{node}_2"
                )

                # Min constraints
                m.addGenConstrMin(
                    min_vars[f"min_{op}_{feature}"][(i, node)],
                    [op_vars[f'{op}_{feature}'][node], y_vars[f"y_{op}_{feature}"][(i, node)]],
                    name=f"min_constr_{op}_{feature}_{i}_{node}"
                )

        for op in discrete_operations:
            for feature in discrete_features:
                # Indicator constraints
                # Min constraints
                m.addGenConstrMin(
                    min_vars_discrete[f"min_{op}_{feature}"][(i, node)],
                    [op_vars[f'{op}_{feature}'][node], rho_vars[f"rho_{op}_{feature}"][(i, node)]],
                    name=f"min_constr_{op}_{feature}_{i}_{node}"
                )

        # Z constraints
        # m.addConstr(
        #     z[i, left_child] <= sum(min_vars[f"min_{op}_{feature}"][(i, node)]
        #                             for op in operations
        #                             for feature in features) + sum(min_vars[f"min_{op}_{feature}"][(i, node)] for op in discrete_operations for feature in discrete_features),
        #     name=f"z_left_constraint_{i}_{node}"
        # )
        #
        # m.addConstr(
        #     z[i, right_child] <= 1 - sum(min_vars[f"min_{op}_{feature}"][(i, node)]
        #                                  for op in operations
        #                                  for feature in features) - sum(min_vars[f"min_{op}_{feature}"][(i, node)] for op in discrete_operations for feature in discrete_features),
        #     name=f"z_right_constraint_1_{i}_{node}"
        # )
        #
        # m.addConstr(
        #     z[i, right_child] <= sum(op_vars[f'{op}_{feature}'][node]
        #                              for op in operations
        #                              for feature in features) + sum(op_vars[f'{op}_{feature}'][node] for op in discrete_operations for feature in discrete_features),
        #     name=f"z_right_constraint_2_{i}_{node}"
        # )

        # Left Child Constraint
        m.addConstr(
            z[i, left_child] <=
            sum(min_vars[f"min_{op}_{feature}"][(i, node)]
                for op in operations
                for feature in features) +
            sum(min_vars_discrete[f"min_{op}_{feature}"][(i, node)]
                for op in discrete_operations
                for feature in discrete_features),
            name=f"z_left_constraint_{i}_{node}"
        )

        # Right Child Constraint (First Part)
        m.addConstr(
            z[i, right_child] <= 1 - (
                    sum(min_vars[f"min_{op}_{feature}"][(i, node)]
                        for op in operations
                        for feature in features) +
                    sum(min_vars_discrete[f"min_{op}_{feature}"][(i, node)]
                        for op in discrete_operations
                        for feature in discrete_features)
            ),
            name=f"z_right_constraint_1_{i}_{node}"
        )

        # Right Child Constraint (Second Part)
        m.addConstr(
            z[i, right_child] <=
            sum(op_vars[f'{op}_{feature}'][node]
                for op in operations
                for feature in features) +
            sum(op_vars[f'{op}_{feature}'][node]
                for op in discrete_operations
                for feature in discrete_features),
            name=f"z_right_constraint_2_{i}_{node}"
        )

lambda_obj = 0
alpha = 0.01

window_sizes = {com: com[1] - com[0] + 1 for com in comb}
window_penalty = 0

m.setObjective(
    (1 - lambda_obj) * quicksum(z_t[i, j]
             for i in range(data_count)
             for j in range(n))
    ,GRB.MAXIMIZE
)

m.optimize()
# Output solution
correct_count = 0
tree_structure = {}
data_results = defaultdict(int) # distribution of data points in each node
if m.status == GRB.OPTIMAL:
    for i in range(data_count):
        if z_s[i].X > 0.5:
            for node_id in range(n):
                if z_t[(i, node_id)].X > 0.5:
                    pred = node_id
                    break
            for l in range(num_labels):
                if w[l][pred].X > 0.5:
                    pred_label = l
                    # if (pred_label == 0 and true_labels[i] <= 0.5) or (pred_label == 1 and true_labels[i] > 0.5): # unnecessary
                    correct_count += 1
                    data_results[node_id] += 1
    print(f"classification Accuracy: labeled data: {correct_count / data_count * 100}%")
    for node_id in range(n):
        restart_loop = 0
        for l in range(num_labels):
            if w[l][node_id].X > 0.5:
                tree_structure[node_id] = l
                print(f"Node {node_id} : label {l}")
                restart_loop = 1

        if not restart_loop:
            for op in operations:
                for feature in features:
                    if op_vars[f'{op}_{feature}'][node_id].X > 0.5:
                        time_data = time_window[node_id]
                        temp = next(com for com in comb if time_data[com].X > 0.5)
                        thresh = threshold[node_id].X
                        comparison = '<=' if op == 'less' else '>='
                        temp_true = f"F{feature}_{temp} {comparison} {thresh}"
                        print(f"Node {node_id} : {temp_true}")
                        tree_structure[node_id] = temp_true
            for op in discrete_operations:
                for feature in discrete_features:
                    if op_vars[f'{op}_{feature}'][node_id].X > 0.5:
                        time_data = time_window[node_id]
                        temp = next(com for com in comb if time_data[com].X > 0.5)
                        comparison = 'belongs to'
                        temp_true = f"F{feature}_{temp} {comparison}'"
                        print(f"Node {node_id} : {temp_true}")
                        tree_structure[node_id] = temp_true
print(data_results)
toc()
