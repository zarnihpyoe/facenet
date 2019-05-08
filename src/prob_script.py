import os, json
from collections import defaultdict
from itertools import permutations
from math import inf, sqrt, factorial, exp
import numpy as np
from random import seed, random
import facenet

seed()

DETECTION_CONSTANT = 0.8
FALSE_ALARM_CONSTANT = 0.8
MISSING_CONSTANT = 0.8

POS_CONSTANT = 0.8

TOTAL_ITERATIONS = 10000
SKIP_FIRST_ITERATIONS = 1000

# this function is kinda works with data/0.3
# def get_index_from_frame(data_dir):
#     classes = [os.path.join(data_dir, p) for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]
#     classes.sort()
#     by_frames = defaultdict(list)
#     i = 0
#     for c in classes:
#         for img_name in sorted(os.listdir(c)):
#             frame = img_name.split('.')[0].split('_')[0]

#             # by_frames[frame].append((i,j))      # i: index in the embeddings, j: label
#             by_frames[frame].append(i)      # i: index in the embeddings
#             i += 1
    
#     return by_frames


def get_index_from_frame(classified_dir):
    dataset = facenet.get_dataset(classified_dir)
    paths, _ = facenet.get_image_paths_and_labels(dataset)
    names = [(path.split('/')[-1]).split('_')[0] for path in paths]
    
    by_frames = defaultdict(list)
    for i, name in enumerate(names):
        by_frames[name].append(i)

    return by_frames
        


def normalize_final_distribution(final_distribution):
    def f(xs):
        return [ x1+x2 for x1, x2 in zip(xs[::2], xs[1::2]) ]
    result_distribution = defaultdict(list)
    for frame in final_distribution.keys():
        result_distribution[frame] = f(final_distribution[frame])
    return result_distribution

# def crappy_test(labels, result_distribution):
#     y_hat = []
#     for frame, dist in result_distribution.items():
#         if get_max_index(dist) == 0 or get_max_index(dist) == 1: y_hat.extend([0,1])
#         elif get_max_index(dist) == 6 or get_max_index(dist) == 7: y_hat.extend([1,0])
#         else: y_hat.extend([-1,-1])
    
#     err = set()
#     for i, (y_,y) in enumerate(zip(y_hat, labels)):
#         if y_ != y: err.add(i//2)
#     print(sorted(list(err)))

def crappy_test(labels, result_distribution):
    err = set()
    for frame, dist in result_distribution.items():
        if not (get_max_index(dist) == 0 or get_max_index(dist) == 1): err.add(frame)
    print(sorted(list(err)))

def get_max_index(xs):
    ans = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[ans]:
            ans = i
    return ans

def fn(data_dir, embeddings, labels, centroids, positions, P_Lt_It_path, P_Lt_path, P_Lt_Lt__1_path, final_distribution_path):
    # DOCS: frames -> { frame_name: [ indexes to use with embeddings or labels ] }
    frames = get_index_from_frame(data_dir)
    # for frame, xs in frames.items():
        # print(frame, xs)
    
    # DOCS: P_Lt_It -> { frame_name: [ probabilities of different configurations based on facenet embeddings ] }
    P_Lt_It = get_prob_from_embeddings(frames, embeddings, labels, centroids)
    with open(P_Lt_It_path, 'w') as f:
        json.dump(P_Lt_It, f)
    print('- done saving P_Lt_It distribution!')

    # DOCS: P_Lt -> { frame_name: [ probabilities of different configurations based on nothing ] }
    P_Lt = get_prob_from_nothing(frames, centroids)
    with open(P_Lt_path, 'w') as f:
        json.dump(P_Lt, f)
    print('- done saving P_Lt distribution!')
    
    # DOCS: P_Lt_Lt__1 -> { frame_name: [ [probabilities of different configurations based on a particular config of previous frame], ... ] }
    P_Lt_Lt__1 = get_prob_from_positions(frames, centroids, positions)
    with open(P_Lt_Lt__1_path, 'w') as f:
        json.dump(P_Lt_Lt__1, f)
    print('- done saving P_Lt_Lt__1 distribution!')

    final_distribution = sampling(P_Lt_It, P_Lt, P_Lt_Lt__1)

    with open(final_distribution_path, 'w') as f:
        json.dump(final_distribution, f)
    print('- done saving final distribution!')

    crappy_test(labels, final_distribution)
    # result_distribution = normalize_final_distribution(final_distribution)
    # crappy_test(labels, result_distribution)
    print('hi')
    return

def sampling(P_Lt_It, P_Lt, P_Lt_Lt__1):
    # S = initialize_sample(get_cumulative_prob(P_Lt))
    S, Sd = initialize_sample_(get_cumulative_prob(P_Lt))
    
    frames = list(sorted(P_Lt.keys()))

    for j in range(TOTAL_ITERATIONS):
        # for each frame...
        for i in range(len(frames)):
            frame = frames[i]
            Pt = [0] * len(P_Lt[frame]) # combined(P_Lt_It[frame], P_Lt[frame], P_Lt_Lt__1[frame][S[prev_frame]]) probabilities for different configurations
            
            if i == 0:
                next_frame = frames[i+1]
                next_frame_config = S[next_frame]

                for c in range(len(P_Lt[frame])):
                    Pt[c] = P_Lt_Lt__1[next_frame][c][next_frame_config] * ( 1 / P_Lt[frame][c] ) * P_Lt_It[frame][c]
                
                S[frame] = sample(get_cumulative_prob_from_raw_probs(Pt))
                if j >= SKIP_FIRST_ITERATIONS: Sd[frame][S[frame]] += 1
                continue
            
            elif i == len(frames)-1:
                prev_frame = frames[i]
                prev_frame_config = S[prev_frame]

                for c in range(len(P_Lt[frame])):
                    Pt[c] = ( P_Lt[prev_frame][prev_frame_config] / P_Lt[frame][c] ) * P_Lt_Lt__1[frame][prev_frame_config][c] * P_Lt_It[frame][c]
                
                S[frame] = sample(get_cumulative_prob_from_raw_probs(Pt))
                if j >= SKIP_FIRST_ITERATIONS: Sd[frame][S[frame]] += 1
                continue

            prev_frame, next_frame = frames[i-1], frames[i+1]
            prev_frame_config, next_frame_config = S[prev_frame], S[next_frame]
            
            # for each possible config... 0..23
            for c in range(len(P_Lt[frame])):
                A = P_Lt_Lt__1[next_frame][c][next_frame_config]
                B = ( P_Lt[prev_frame][prev_frame_config] / P_Lt[frame][c] )
                C = P_Lt_Lt__1[frame][prev_frame_config][c]
                D = P_Lt_It[frame][c]
                # Pt[c] = P_Lt_Lt__1[next_frame][c][next_frame_config] * ( P_Lt[prev_frame][prev_frame_config] / P_Lt[frame][c] ) * P_Lt_Lt__1[frame][prev_frame_config][c] * P_Lt_It[frame][c]
                Pt[c] = A * B * C * D
            
            S[frame] = sample(get_cumulative_prob_from_raw_probs(Pt))
            if j >= SKIP_FIRST_ITERATIONS: Sd[frame][S[frame]] += 1

    final_distribution = compute_distribution_from_sampling(Sd)

    print('- done computing final distribution')
    return final_distribution


def initialize_sample(C_P_Lt):
    S = defaultdict(int)
    for frame, probs in C_P_Lt.items():
        S[frame] = sample(probs)
    return S
def initialize_sample_(C_P_Lt):
    S = defaultdict(int)
    Sd = defaultdict(list)
    for frame, probs in C_P_Lt.items():
        S[frame] = sample(probs)
        Sd[frame] = [0] * len(probs)
        # Sd[frame][S[frame]] += 1
    return S, Sd

# probs: Cumulative Probability Distribution (array of probabilities)
def sample(probs):
    rdn = random()
    # print(rdn)
    for i, prob in enumerate(probs):
        if rdn <= prob: return i
    return len(probs) - 1

def compute_distribution_from_sampling(Sd):
    final_distribution = defaultdict(list)
    for frame, freqs in Sd.items():
        total_freq = sum(freqs)
        final_distribution[frame] = [ freq/total_freq for freq in freqs ]
    return final_distribution

def get_prob_from_embeddings(frames, embeddings, labels, centroids):
    n = centroids.shape[1]
    prob_dict = defaultdict(list)
    # for each frame, ...
    for frame, bbs in frames.items():       # bbs: [index of embedding, ...]
        # print(frame)
        m = len(bbs)

        # for each configuration out of k configurations: compute the cost, and put it in the 'costs' array
        costs = [0] * factorial(m+n)
        for k, xs in enumerate(permutations(range(m+n))):
            # consider i -> xs[i];
            for i,j in enumerate(xs):
                if i < n:  # person case
                    if j < m:   # person -> detected
                        costs[k] += distance(centroids[:, i], embeddings[:, bbs[j]])
                    else:       # person -> missed
                        costs[k] += MISSING_CONSTANT
                
                else:   # false alarm case
                    if j < m:   # false alarm -> detected
                        costs[k] += FALSE_ALARM_CONSTANT
                    else:       # false alarm -> missed
                        costs[k] += 0
            # print(costs[k])
        
        prob_dict[frame] = get_prob_from_raw_costs(costs)
        # print(frame, ':', cost_dict[frame])
    
    return prob_dict
    # return get_prob_from_normalized_costs(cost_dict)

# we need frames to get 'm'; we need centroids to get 'n'
# other than that, we are not using any values from them
# Prorior
def get_prob_from_nothing(frames, centroids):
    n = centroids.shape[1]
    prob_dict = defaultdict(list)
    
    # for each frame, ...
    for frame, bbs in frames.items():       # bbs: [index of embedding, ...]
        m = len(bbs)
        costs = [0] * factorial(m+n)
        for k, xs in enumerate(permutations(range(m+n))):
            for i,j in enumerate(xs):
                if i < n:  # person case
                    if j < m:   # person -> detected
                        costs[k] += DETECTION_CONSTANT
                    else:       # person -> missed
                        costs[k] += MISSING_CONSTANT

                else:   # false alarm case
                    if j < m:   # false alarm -> detected
                        costs[k] += FALSE_ALARM_CONSTANT
                    else:       # false alarm -> missed
                        costs[k] += 0
                
            # costs[k] = exp(costs[k])
        prob_dict[frame] = get_prob_from_raw_costs(costs)
        
        # total_cost = sum(costs)
        # cost_dict[frame] = [ cost/total_cost for cost in costs ]
        # print(frame, ':', cost_dict[frame])
    return prob_dict
    # return get_prob_from_normalized_costs(cost_dict)

def get_prob_from_positions(frames, centroids, positions):
    n = centroids.shape[1]
    frame_keys = list(sorted(frames.keys()))
    
    probs_dict = defaultdict(list)
    
    # from previous frame
    # for each frame
    for a in range(1, len(frame_keys)):
        prev_frame_bbs = frames[frame_keys[a-1]]
        curr_frame_bbs = frames[frame_keys[a]]

        for k, xs in enumerate(permutations(range(len(prev_frame_bbs)+n))):
            prev_frame_positions = []
            for i,j in enumerate(xs):
                if i < n:
                    if j < len(prev_frame_bbs):     # person -> detected
                        prev_frame_positions.append(positions[:, prev_frame_bbs[j]])
                    else:       # person -> missing
                        prev_frame_positions.append([inf,inf])
                # TODO: things to consider here for FA -> Detection and FA -> Missing ...
                else:
                    prev_frame_positions.append([inf,inf])
            
            # print(prev_frame_positions)
            costs = [0] * factorial(len(curr_frame_bbs)+n)
            for l, ys in enumerate(permutations(range(len(curr_frame_bbs)+n))):
                curr_frame_positions = []
                for p,q in enumerate(ys):
                    if p < n:
                        if q < len(curr_frame_bbs):     # person -> detected
                            curr_frame_positions.append(positions[:, curr_frame_bbs[q]])
                        else:       # person -> missing
                            curr_frame_positions.append([inf,inf])
                    # TODO: things to consider here for FA -> Detection and FA -> Missing ...
                    else:
                        curr_frame_positions.append([inf,inf])

                # compute the distance between positions
                for prev_pos, curr_pos in zip(prev_frame_positions, curr_frame_positions):
                    dist = pos_distance(prev_pos, curr_pos)
                    costs[l] += dist
            
            # print(costs)
            probs_dict[frame_keys[a]].append(get_prob_from_raw_costs(costs))
        
    return probs_dict


def distance(embedding1, embedding2):
    diff = embedding1 - embedding2
    return diff.dot(diff)

# return squared distance between pos1 and pos2
# return 0 if the position is at inf
def pos_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    if x1 == inf or x2 == inf or y1 == inf or y2 == inf:
        return POS_CONSTANT
    return (x1-x2)**2 + (y1-y2)**2
    
# raw cost_array -> normalized prob_array
def get_prob_from_raw_costs(costs):
    probs = [ exp(1/cost) for cost in costs ]
    total_prob = sum(probs)
    probs = [ prob/total_prob for prob in probs ]
    # generating cumulative probs
    # for i in range(1, len(probs)):
    #     probs[i] += probs[i-1]
    return probs

def get_cumulative_prob_from_raw_probs(probs):
    total_prob = sum(probs)
    normalized_probs = [ prob/total_prob for prob in probs ]
    for i in range(1, len(normalized_probs)):
        normalized_probs[i] += normalized_probs[i-1]
    return normalized_probs


# prob_dict -> c_prob_dict
def get_cumulative_prob(prob_dict):
    c_prob_dict = defaultdict(list)
    for frame, probs in prob_dict.items():
        c_prob_dict[frame] = probs[:]
        for i in range(1, len(probs)):
            c_prob_dict[frame][i] += c_prob_dict[frame][i-1]

    return c_prob_dict

# def get_min_index(xs):
#     ans = 0
#     for i in range(1, len(xs)):
#         if xs[i] < xs[ans]:
#             ans = i
#     return ans

    
# data_dir = '/Users/zarnihpyoe/wpi/mqp/data/0.2(subset)'
# embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.2(subset)/embeddings.npy')
# centroids = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.2(subset)/centroids_embeddings.npy')
# positions = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.2(subset)/face_positions.npy')

# data_dir = '/Users/zarnihpyoe/wpi/mqp/data/0.3(subset)'
# embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3(subset)/embeddings.npy')
# centroids = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3(subset)/centroids_embeddings.npy')
# positions = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3(subset)/face_positions.npy')

# data_dir = '/Users/zarnihpyoe/wpi/mqp/data/0.3'
# embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3/embeddings.npy')
# centroids = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3/centroids_embeddings.npy')
# positions = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3/face_positions.npy')


# data_dir = '/Users/zarnihpyoe/wpi/mqp/data/0.3(midsubset)'
# embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3(midsubset)/embeddings.npy')
# centroids = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3(midsubset)/centroids_embeddings.npy')
# positions = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3(midsubset)/face_positions.npy')

# data_dir = '/Users/zarnihpyoe/wpi/mqp/data4/raw/classified'
# embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data4/0.1/embeddings.npy')
# centroids = np.load('/Users/zarnihpyoe/wpi/mqp/data4/0.1/centroids_embeddings.npy')
# positions = np.load('/Users/zarnihpyoe/wpi/mqp/data4/0.1/face_positions.npy')
# P_Lt_It_path = '/Users/zarnihpyoe/wpi/mqp/data4/0.1/P_Lt_It_distribution.json'
# P_Lt_path = '/Users/zarnihpyoe/wpi/mqp/data4/0.1/P_Lt_distribution.json'
# P_Lt_Lt__1_path = '/Users/zarnihpyoe/wpi/mqp/data4/0.1/P_Lt_Lt__1_distribution.json'
# final_distribution_path = '/Users/zarnihpyoe/wpi/mqp/data4/0.1/final_distribution.json'

data_dir = '/Users/zarnihpyoe/wpi/mqp/data5/classified'
embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data5/embeddings.npy')
centroids = np.load('/Users/zarnihpyoe/wpi/mqp/data5/centroids_embeddings.npy')
positions = np.load('/Users/zarnihpyoe/wpi/mqp/data5/face_positions.npy')
P_Lt_It_path = '/Users/zarnihpyoe/wpi/mqp/data5/P_Lt_It_distribution.json'
P_Lt_path = '/Users/zarnihpyoe/wpi/mqp/data5/P_Lt_distribution.json'
P_Lt_Lt__1_path = '/Users/zarnihpyoe/wpi/mqp/data5/P_Lt_Lt__1_distribution.json'
final_distribution_path = '/Users/zarnihpyoe/wpi/mqp/data5/final_distribution.json'


fn(data_dir, embeddings, labels, centroids, positions, P_Lt_It_path, P_Lt_path, P_Lt_Lt__1_path, final_distribution_path)

# def testing_data4(final_distribution_path):
#     fd = None
#     with open(final_distribution_path) as f:
#         fd = json.load(f)
    
#     print('list of wrong frames')
#     for frame, distrbution in fd.items():
#         i = get_max_index(distrbution)
#         if i not in [0,1]: print('frame: {}\tdistribution: {}'.format(frame, distrbution))
#     print('testing done!!!')


# testing_data4(final_distribution_path)

# for k, v in get_index_from_frame(data_dir).items():
#     print(k,v)


# crappy_test(labels, final_distribution)