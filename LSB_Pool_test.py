import stego
import cv2
import time
import numpy as np
import bisect

def sort_along_0(array_in):
    array_shape = list(np.shape(array_in))
    second_axis_size = 1
    for i in array_shape[1:]:
        second_axis_size *= i
    array_reshape = array_in.reshape((array_shape[0], second_axis_size))
    array_reshape_t = np.transpose(array_reshape)
    lex_index = np.lexsort(array_reshape_t[::-1, :])
    array_reshape_sorted = array_reshape[lex_index]
    return array_reshape_sorted.reshape(tuple(array_shape))

def sort_along_axis(array_in, axis=None, debug=None):
    dimensions = len(list(np.shape(array_in)))
    if debug: print("input shape:", np.shape(array_in))
    if debug is None:
        debug = False
    if (axis is None) or (axis >= dimensions) or (axis == 0):
        return sort_along_0(array_in)
    array_shape = list(np.shape(array_in))
    new_shape = array_shape[axis-1:]
    new_shape[0] = 1
    for i in array_shape[:axis]:
        new_shape[0] *= i
    array_reshape = array_in.reshape(tuple(new_shape))
    if debug: print("array_reshape:\n", array_reshape)
    for i in range(len(array_reshape)):
        array_reshape[i] = sort_along_0(array_reshape[i])
    return array_reshape.reshape(tuple(array_shape))

def convert(point_set, img_shape):
    if not type(img_shape) is list:
        img_shape = list(img_shape)

def reduce_pairing_order_v5(pairing, iterations=None):
    points_are_raw = False
    if iterations is None:
        iterations = 50
    if len(list(np.shape(pairing))) >= 3:
        points_are_raw = True
        points_row, points_col = pairing[:, :, 0], pairing[:, :, 1]
        max_final = max(np.max(points_row) + 1, np.max(points_col) + 1)
        pairing = np.add(np.multiply(points_row, max_final), points_col).astype('int64')
        pairing = np.sort(pairing, axis=1)
        pairing = sort_along_0(pairing)
        slice_1, slice_2 = pairing[:, 0], pairing[:, 1]
        slice_1_compare, slice_2_compare = np.concatenate(([slice_1[0] - 1], slice_1[:-1])), np.concatenate(
            ([slice_2[0] - 1], slice_2[:-1]))
        slice_1_check, slice_2_check = np.not_equal(slice_1, slice_1_compare), np.not_equal(slice_2, slice_2_compare)
        final_check = np.logical_or(slice_1_check, slice_2_check)
        pairing = pairing[np.where(final_check == True)]
    all_ints_sorted = np.unique(np.ndarray.flatten(np.array(pairing)))
    index_match = np.arange(max(all_ints_sorted)+1)
    index_match[all_ints_sorted] = np.arange(len(all_ints_sorted))
    #index array such that all_ints_sorted[I[pair_val]] = pairval
    #print("all ints sorted")
    #print(all_ints_sorted)
    all_ints_sorted = np.array([all_ints_sorted, np.arange(len(all_ints_sorted))]) #pairs, then their chain number.
    pairing_t = np.transpose(pairing)
    old_assignments = all_ints_sorted[1]
    new_assignments = old_assignments

    for i in range(iterations):
        i_1, i_2 = index_match[pairing_t[0]], index_match[pairing_t[1]]
        min_set = np.maximum(old_assignments[i_1], old_assignments[i_2])
        new_assignments[i_1] = min_set
        min_set = np.maximum(new_assignments[i_1], new_assignments[i_2])
        new_assignments[i_2] = min_set
        if i != iterations-1:
            old_assignments = new_assignments
        #count_unique_groups = sum(np.not_equal(new_assignments,
        #                                       np.concatenate(([new_assignments[0]-1],
        #                                                      new_assignments[:-1]))))
        #print("Unique groups:", count_unique_groups)


    pairs_chain_assigned = np.transpose(np.array([new_assignments, all_ints_sorted[0]]))
    pairs_chain_assigned = np.transpose(sort_along_0(pairs_chain_assigned))
    pairs_chain_assigned = np.array([np.floor_divide(pairs_chain_assigned[1], max_final),
                                     np.mod(pairs_chain_assigned[1], max_final),
                                     pairs_chain_assigned[0]])
    pairs_chain_assigned = np.transpose(pairs_chain_assigned)
    chain_index = pairs_chain_assigned[:,2]
    indexing_chains = np.not_equal(chain_index, np.concatenate((chain_index[1:], [chain_index[-1]+1])))
    chain_markers = np.add(np.where(indexing_chains), 1)
    chain_markers = np.concatenate(([0], chain_markers[0]))

    chains_return = []

    for i in range(len(chain_markers)-1):
        temp_chain = pairs_chain_assigned[chain_markers[i]:chain_markers[i+1]]
        temp_chain = temp_chain[:,[0,1]]
        chains_return.append(temp_chain)

    return np.array(chains_return, dtype=object)

def faster_unique_2d(in_array, sorted=None):
    if sorted is None:
        sorted = False
    if not sorted:
        in_array = sort_along_0(in_array)
    if len(list(np.shape(in_array))) == 1:
        check_in = (in_array != np.concatenate(([in_array[0] - 1], in_array[:-1])))
        index_unique = np.where(check_in)
        return in_array[index_unique]
    elif len(list(np.shape(in_array))) == 2:
        check_in = np.zeros_like(in_array).astype('bool')
        for i in range(len(in_array[0])):
            check_in[:, i] = (in_array[:, i] != np.concatenate(([in_array[0][i] - 1], in_array[:-1, i])))
        check_in = np.sum(check_in.astype('int'), axis=1)
        check_in = tuple(check_in)
        index_unique = np.where(np.greater(check_in, 0))
        return in_array[index_unique]
    else:
        raise ValueError


def make_pairs_from_overwrite(list_1, list_2):
    """
    Takes two 1 dimensional lists of same size,
    creates a pairing list of differences between the lists.
    i.e. [1, 2, 3, 4], [1, 5, 2, 6] are inputs,
    it returns [[2, 5],  [4, 6]] as an output.
    it attempts to make these into a chain, however.
    """
    if np.shape(list_1) != np.shape(list_2):
        raise ValueError
    index_different = np.where(list_1 != list_2)
    pairs = np.transpose([list_1[index_different], list_2[index_different]])
    #chains = reduce_pairing_order_v4(pairs)
    #return chains
    return pairs


def attempt_reassignment_from_association(assignments, old_assignments):
    assignment_pairs = make_pairs_from_overwrite(old_assignments, assignments)
    assignment_chains = reduce_pairing_order_v4(assignment_pairs)
    a_argsort = assignments.argsort()
    assignments = assignments[a_argsort]

    revert_order_index = np.arange(len(assignments))
    revert_order_index = revert_order_index[a_argsort]

    unique_assignments = np.unique(assignments)
    assignment_index = np.array(np.where(assignments != np.concatenate((assignments[1:], [assignments[-1] + 1]))))
    assignment_index = np.add(assignment_index, 1)
    assignment_index = assignment_index[0]
    assignment_index = np.concatenate(([0], assignment_index))

    for chain in assignment_chains:
        print("chain:", chain)
        for i in range(1, len(chain)):
            unique_index = bisect.bisect_left(unique_assignments, chain[i])
            i_1, i_2 = assignment_index[unique_index], assignment_index[unique_index + 1]
            assignments[i_1:i_2] = np.full_like(assignments[i_1:i_2], chain[0])

    assignments = assignments[revert_order_index]

    return assignments

def reduce_pairing_backend_v2(pairing, pairings_are_sorted=None, pairings_are_unique=None, return_chains=None,
                              recursion_allowed=None):
    reduction_complete = False
    if recursion_allowed is None:
        recursion_allowed = True
    if return_chains is None:
        return_chains = True
    if pairings_are_unique is None:
        pairings_are_unique = False
    if pairings_are_sorted is None:
        pairings_are_sorted = False
    if not pairings_are_unique:
        pairings_are_sorted = False
    if not pairings_are_sorted:
        pairing = np.sort(pairing, axis=1)
    if not pairings_are_unique:
        pairing = faster_unique_2d(pairing, sorted=pairings_are_sorted)
    all_ints_sorted = faster_unique_2d(np.ndarray.flatten(pairing))
    points_are_assigned = np.zeros_like(all_ints_sorted).astype('bool')
    group_assignments = np.zeros_like(all_ints_sorted)
    current_max_group = 1
    p1_i, p2_i = bisect.bisect_left(all_ints_sorted, pairing[0][0]), bisect.bisect_left(all_ints_sorted, pairing[0][1])
    points_are_assigned[p1_i] = True
    points_are_assigned[p2_i] = True
    group_assignments[p1_i], group_assignments[p2_i] = 1, 1
    group_assignment_overlaps = []
    for i, pair in enumerate(pairing[1:]):
        p1_i, p2_i = bisect.bisect_left(all_ints_sorted, pair[0]), bisect.bisect_left(all_ints_sorted, pair[1])
        c1, c2 = points_are_assigned[p1_i], points_are_assigned[p2_i]
        if c1 and (not c2):
            points_are_assigned[p2_i] = True
            group_assignments[p2_i] = group_assignments[p1_i]
        elif c2 and (not c1):
            points_are_assigned[p1_i] = True
            group_assignments[p1_i] = group_assignments[p2_i]
        elif (not c1) and (not c2):
            current_max_group += 1
            group_assignments[p1_i], group_assignments[p2_i] = current_max_group, current_max_group
            points_are_assigned[p1_i], points_are_assigned[p2_i] = True, True
        else:
            if group_assignments[p1_i] != group_assignments[p2_i]:
                group_assignment_overlaps.append([group_assignments[p1_i], group_assignments[p2_i]])

    group_assignment_overlaps = np.sort(group_assignment_overlaps, axis=1)
    group_assignment_overlaps = faster_unique_2d(group_assignment_overlaps)

    group_assignment_chains = reduce_pairing_order_v4(group_assignment_overlaps)


    print("Done! Max chain index:", current_max_group)
    print("Sum of 0s:", sum(np.equal(group_assignments, 0).astype('int')))
    print("Group overlaps (", len(group_assignment_overlaps), " total ):")
    print(group_assignment_overlaps)

    print("Group overlap chains:")
    print(group_assignment_chains)

    if not return_chains:
        return np.array([all_ints_sorted, group_assignments])

    point_group_list = np.transpose([group_assignments, all_ints_sorted])
    point_group_list = sort_along_0(point_group_list)
    groups_list = point_group_list[:,0]

    chain_markers = np.not_equal(groups_list, np.concatenate((groups_list[1:], [groups_list[-1] + 1])))
    chain_markers = np.add(np.where(chain_markers), 1)
    chain_markers = np.concatenate(([0], chain_markers[0]))

    chains_return = []

    unique_groups = faster_unique_2d(groups_list, sorted=True)

    for chain in group_assignment_chains:
        true_assignment = chain[-1]
        for group in chain[:-1]:
            index_group = bisect.bisect_left(unique_groups, group)
            b1, b2 = chain_markers[index_group], chain_markers[index_group+1]
            point_group_list[b1:b2,0] = true_assignment

    point_group_list = sort_along_0(point_group_list)
    groups_list = point_group_list[:, 0]
    point_list_sorted_by_group = point_group_list[:, 1]

    chain_markers = np.not_equal(groups_list, np.concatenate((groups_list[1:], [groups_list[-1] + 1])))
    chain_markers = np.add(np.where(chain_markers), 1)
    chain_markers = np.concatenate(([0], chain_markers[0]))


    for i in range(len(chain_markers) - 1):
        temp_chain = point_list_sorted_by_group[chain_markers[i]:chain_markers[i + 1]]
        chains_return.append(temp_chain)
    return np.array(chains_return, dtype=object)

def reduce_pairing_backend_v3(pairing, pairings_are_sorted=None, pairings_are_unique=None, return_chains=None,
                              recursion_allowed=None):
    reduction_complete = False
    if recursion_allowed is None:
        recursion_allowed = True
    if return_chains is None:
        return_chains = True
    if pairings_are_unique is None:
        pairings_are_unique = False
    if pairings_are_sorted is None:
        pairings_are_sorted = False
    if not pairings_are_unique:
        pairings_are_sorted = False
    if not pairings_are_sorted:
        pairing = np.sort(pairing, axis=1)
    if not pairings_are_unique:
        pairing = faster_unique_2d(pairing, sorted=pairings_are_sorted)
    all_ints_sorted = faster_unique_2d(np.ndarray.flatten(pairing))

    #all_ints_sorted := sorted set of all unique elements in the pair set
    reference_array = np.swapaxes([np.arange(len(all_ints_sorted)), np.ones_like(all_ints_sorted)], 0, 1)
    pairing_dims_get = np.shape(pairing)
    new_pairs = np.ndarray.flatten(pairing)

    print("pairing_flattened")

    new_pairs = np.searchsorted(all_ints_sorted, new_pairs)
    new_pairs = np.reshape(new_pairs, pairing_dims_get)

    loop_flag = True
    iteration_count = 0
    active_pairs = new_pairs

    assignments_1 = reference_array[active_pairs[:,0],0]
    assignments_2 = reference_array[active_pairs[:,1],0]
    

    while (loop_flag):
        count_operating_pixels = sum(reference_array[:,1])
        print("operating pixel pool -> ", count_operating_pixels)
        print("updating reference, iteration",iteration_count)
        old_reference_hash = hash(reference_array.tobytes())

        is_reference_1 = np.equal(reference_array[assignments_1, 1], 0)
        is_reference_2 = np.equal(reference_array[assignments_2, 1], 0)

        #While loop below gets the true references group assignment of the entire array
        #retains this info through loops so as not to repeat the process of finding assigned groups
        assignments_complete = False
        while (not assignments_complete):
            update_indices_1 = np.where(is_reference_1)
            update_indices_2 = np.where(is_reference_2)
            if (update_indices_1[0].size == 0 and update_indices_2[0].size == 0):
                assignments_complete = True
                continue
            assignments_1[update_indices_1] = reference_array[assignments_1[update_indices_1],0]
            is_reference_1[update_indices_1] = np.equal(reference_array[assignments_1[update_indices_1],1], 0)
            assignments_2[update_indices_2] = reference_array[assignments_2[update_indices_2],0]
            is_reference_2[update_indices_2] = np.equal(reference_array[assignments_2[update_indices_2],1], 0)

        #updates right side of pairs where left assignment is less than right assignment
        less_indices = np.where(np.less(assignments_1, assignments_2))
        reference_array[assignments_2[less_indices], 0] = assignments_1[less_indices]
        reference_array[assignments_2[less_indices], 1] = np.zeros_like(assignments_2[less_indices])        
        active_pairs[less_indices, 1][0] = reference_array[active_pairs[less_indices, 1][0], 0] 

        #updates left side of pairs where left assignment is greater than right assignment
        greater_indices = np.where(np.greater(assignments_1, assignments_2))
        reference_array[assignments_1[greater_indices], 0] = assignments_2[greater_indices]
        reference_array[assignments_1[greater_indices], 1] = np.zeros_like(assignments_1[greater_indices])
        active_pairs[greater_indices, 0][0] = reference_array[active_pairs[greater_indices, 0][0], 0]
        #the loop below updates reference_array values to point to true root
        #for a given reference sequence of length n, it takes roughly log2(n) iterations to collapse
        break_flag = False
        while (not break_flag):
            u_ind_1 = np.where(np.equal(reference_array[:,1], 0))[0]
            u_ind_2 = np.where(np.equal(reference_array[reference_array[u_ind_1,0],1], 0))[0]
            if (u_ind_2.size == 0):
                break_flag = True
            reference_array[u_ind_1[u_ind_2],0] = reference_array[reference_array[u_ind_1[u_ind_2],0], 0]
        #only at this point is the shape of active pairs reduced
        pairs_to_keep_indices = np.where(np.not_equal(active_pairs[:,0], active_pairs[:,1]))
        active_pairs = active_pairs[pairs_to_keep_indices]
        assignments_1 = assignments_1[pairs_to_keep_indices]
        assignments_2 = assignments_2[pairs_to_keep_indices]
        new_reference_hash = hash(reference_array.tobytes())

        if (old_reference_hash == new_reference_hash):
            print("identical hashes; breaking loop")
            loop_flag = False

        iteration_count += 1
        if (iteration_count >= 200):
            loop_flag = False
    

    new_reference = np.copy(reference_array)

    break_flag = False
    while (not break_flag):
        target_indices = np.where(np.equal(new_reference[:,1],0))[0]
        print("target indices\n",target_indices)
        if (target_indices.size == 0):
            break_flag = True
        new_reference[target_indices] = new_reference[new_reference[target_indices,0]]

    group_assignments = new_reference[:,0]


    if not return_chains:
        return np.array([all_ints_sorted, group_assignments])

    point_group_list = np.transpose([group_assignments, all_ints_sorted])
    point_group_list = sort_along_0(point_group_list)
    groups_list = point_group_list[:,0]

    chain_markers = np.not_equal(groups_list, np.concatenate((groups_list[1:], [groups_list[-1] + 1])))
    chain_markers = np.add(np.where(chain_markers), 1)
    chain_markers = np.concatenate(([0], chain_markers[0]))

    chains_return = []

    for i in range(len(chain_markers) - 1):
        temp_chain = point_group_list[chain_markers[i]:chain_markers[i + 1], 1]
        chains_return.append(temp_chain)
    return np.array(chains_return, dtype=object)

def reduce_pairing_backend(pairing, pairings_are_sorted=None, pairings_are_unique=None, iterations=None,
                           return_chains=None):
    reduction_complete = False
    if return_chains is None:
        return_chains = True
    if iterations is None:
        iterations = 10
    if pairings_are_unique is None:
        pairings_are_unique = False
    if pairings_are_sorted is None:
        pairings_are_sorted = False
    if not pairings_are_unique:
        pairings_are_sorted = False
    if not pairings_are_sorted:
        pairing = np.sort(pairing, axis=1)
    if not pairings_are_unique:
        pairing = faster_unique_2d(pairing, sorted=pairings_are_sorted)
    all_ints_sorted = np.unique(np.ndarray.flatten(np.array(pairing)))
    index_match = np.arange(max(all_ints_sorted) + 1)
    index_match[all_ints_sorted] = np.arange(len(all_ints_sorted))
    all_ints_sorted = np.array([all_ints_sorted, np.arange(len(all_ints_sorted))])
    pairing_t = np.transpose(pairing)
    old_assignments = all_ints_sorted[1]
    new_assignments = np.copy(old_assignments)

    unique_groups = []

    for i in range(iterations):
        #assignments are each point as a number, unique and sorted
        #pairing_t is a n x 2 array of each pair unidirectional and sorted

        i_1, i_2 = index_match[pairing_t[0]], index_match[pairing_t[1]]
        min_set = np.maximum(old_assignments[i_1], old_assignments[i_2])
        new_assignments[i_1] = min_set
        min_set = np.maximum(new_assignments[i_1], new_assignments[i_2])
        new_assignments[i_2] = min_set

        unique_groups.append([i, len(faster_unique_2d(new_assignments))])
        print("Iteration:", unique_groups[i])

        if hash(new_assignments.tobytes()) == hash(old_assignments.tobytes()):
            reduction_complete = True
            break
        if i < iterations-1:
            old_assignments = np.copy(new_assignments)

    print("Final old hash:", hash(old_assignments.tobytes()))
    print("Final new hash:", hash(new_assignments.tobytes()))

    if not reduction_complete and True:
        write_name = 'test_assignments.txt'
        test_assignments = attempt_reassignment_from_association(new_assignments, old_assignments)
        with open('test_assignments.txt', 'w') as f:
            for i in range(len(test_assignments)):
                f.write(str(all_ints_sorted[0][i])+'->'+str(test_assignments[i]))
                f.write('\n')


    #if reduction_complete:
        #with open('perfect_assignments.txt', 'w') as f:
            #for i in range(len(new_assignments)):
                #f.write(str(all_ints_sorted[0][i])+'->'+str(new_assignments[i]))
                #f.write('\n')


    if not return_chains:
        return np.transpose(np.array([all_ints_sorted[0], new_assignments]))

    print("1")
    elements_chain_assigned = np.transpose(np.array([new_assignments, all_ints_sorted[0]]))
    elements_chain_assigned = sort_along_0(elements_chain_assigned)





    chain_index = elements_chain_assigned[:,0]
    indexing_chains = np.not_equal(chain_index, np.concatenate((chain_index[1:], [chain_index[-1]+1])))
    chain_markers = np.add(np.where(indexing_chains), 1)
    chain_markers = np.concatenate(([0], chain_markers[0]))

    chains_return = []
    for i in range(len(chain_markers)-1):
        temp_chain = elements_chain_assigned[chain_markers[i]:chain_markers[i+1]]
        temp_chain = temp_chain[:,[0,1]]
        chains_return.append(temp_chain)
    print("2")
    return np.array(chains_return, dtype=object)

def reduce_pairing_order_v7(pairing, iterations=None, pairings_are_sorted=None, pairings_are_unique=None):
    points_are_tuples = False
    if len(list(np.shape(pairing))) >= 3:
        points_are_tuples = True
        points_row, points_col = pairing[:, :, 0], pairing[:, :, 1]
        max_final = max(np.max(points_row) + 1, np.max(points_col) + 1)
        pairing = np.add(np.multiply(points_row, max_final), points_col).astype('int64')

    #reduced_pairing = reduce_pairing_backend(pairing, pairings_are_sorted=pairings_are_sorted,
    #                                         pairings_are_unique=pairings_are_unique, iterations=iterations,
    #                                         return_chains=return_chains)

    reduced_pairing = reduce_pairing_backend_v3(pairing, pairings_are_sorted=pairings_are_sorted,
                                                pairings_are_unique=pairings_are_unique,
                                                return_chains=True)

    if not points_are_tuples:
        return reduced_pairing

    for i, chain in enumerate(reduced_pairing):
        reduced_pairing[i] = np.transpose([np.floor_divide(chain, max_final), np.mod(chain, max_final)])

    return np.array(reduced_pairing, dtype=object)

def reduce_pairing_order_v6(pairing, iterations=None):
    points_are_raw = False
    reduction_complete = False
    if iterations is None:
        iterations = 50
    if len(list(np.shape(pairing))) >= 3:
        points_are_raw = True
        points_row, points_col = pairing[:, :, 0], pairing[:, :, 1]
        max_final = max(np.max(points_row) + 1, np.max(points_col) + 1)
        pairing = np.add(np.multiply(points_row, max_final), points_col).astype('int64')
        pairing = np.sort(pairing, axis=1)
        pairing = sort_along_0(pairing)
        slice_1, slice_2 = pairing[:, 0], pairing[:, 1]
        slice_1_compare, slice_2_compare = np.concatenate(([slice_1[0] - 1], slice_1[:-1])), np.concatenate(
            ([slice_2[0] - 1], slice_2[:-1]))
        slice_1_check, slice_2_check = np.not_equal(slice_1, slice_1_compare), np.not_equal(slice_2, slice_2_compare)
        final_check = np.logical_or(slice_1_check, slice_2_check)
        pairing = pairing[np.where(final_check == True)]
    all_ints_sorted = np.unique(np.ndarray.flatten(np.array(pairing)))
    index_match = np.arange(max(all_ints_sorted)+1)
    index_match[all_ints_sorted] = np.arange(len(all_ints_sorted))
    all_ints_sorted = np.array([all_ints_sorted, np.arange(len(all_ints_sorted))]) #pairs, then their chain number.
    pairing_t = np.transpose(pairing)
    old_assignments = all_ints_sorted[1]
    new_assignments = np.copy(old_assignments)

    map_unique_groups = []

    for i in range(iterations):
        i_1, i_2 = index_match[pairing_t[0]], index_match[pairing_t[1]]
        min_set = np.maximum(old_assignments[i_1], old_assignments[i_2])
        new_assignments[i_1] = min_set
        min_set = np.maximum(new_assignments[i_1], new_assignments[i_2])
        new_assignments[i_2] = min_set
        print("Iteration", i, "Assignments hash:", hash(new_assignments.tobytes()))
        if hash(new_assignments.tobytes()) == hash(old_assignments.tobytes()):
            print("red")
            reduction_complete = True
            break
        if i < iterations-1:
            old_assignments = np.copy(new_assignments)

    print("Final old hash:", hash(old_assignments.tobytes()))
    print("Final new hash:", hash(new_assignments.tobytes()))

    if not reduction_complete:
        new_pairs = make_pairs_from_overwrite(old_assignments, new_assignments)

    pairs_chain_assigned = np.transpose(np.array([new_assignments, all_ints_sorted[0]]))
    pairs_chain_assigned = np.transpose(sort_along_0(pairs_chain_assigned))
    pairs_chain_assigned = np.array([np.floor_divide(pairs_chain_assigned[1], max_final),
                                     np.mod(pairs_chain_assigned[1], max_final),
                                     pairs_chain_assigned[0]])
    pairs_chain_assigned = np.transpose(pairs_chain_assigned)
    chain_index = pairs_chain_assigned[:,2]
    indexing_chains = np.not_equal(chain_index, np.concatenate((chain_index[1:], [chain_index[-1]+1])))
    chain_markers = np.add(np.where(indexing_chains), 1)
    chain_markers = np.concatenate(([0], chain_markers[0]))

    chains_return = []

    for i in range(len(chain_markers)-1):
        temp_chain = pairs_chain_assigned[chain_markers[i]:chain_markers[i+1]]
        temp_chain = temp_chain[:,[0,1]]
        chains_return.append(temp_chain)

    return np.array(chains_return, dtype=object)

def reduce_pairing_order_v4(pairing): #time complexity is O(n*log(n)), extremely fast
    points_are_raw = False
    if len(list(np.shape(pairing))) >= 3:
        #print("Points are raw")
        #print("Point input:\n", pairing)
        points_are_raw = True
        points_row, points_col = pairing[:,:,0], pairing[:,:,1]
        max_final = max(np.max(points_row) + 1, np.max(points_col) + 1)
        pairing = np.add(np.multiply(points_row, max_final), points_col).astype('int64')
        pairing = np.sort(pairing, axis=1)
        time_get = time.time()
        pairing = sort_along_0(pairing)
        time_get = time.time()-time_get
        print("sorted in:", time_get)
        slice_1, slice_2 = pairing[:,0], pairing[:,1]
        slice_1_compare, slice_2_compare = np.concatenate(([slice_1[0]-1], slice_1[:-1])), np.concatenate(([slice_2[0]-1], slice_2[:-1]))
        slice_1_check, slice_2_check = np.not_equal(slice_1, slice_1_compare), np.not_equal(slice_2, slice_2_compare)
        final_check = np.logical_or(slice_1_check, slice_2_check)
        pairing = pairing[np.where(final_check == True)]
        print("pairing size:", np.shape(pairing))
        print("Done with this")

    pairing_forward = pairing
    pairing_backward = sort_along_0(pairing[:,[1,0]])

    all_ints_sorted = np.sort(np.unique(np.ndarray.flatten(np.array(pairing))))
    #print(all_ints_sorted)
    #print("1")
    chain_indices = np.zeros_like(all_ints_sorted)
    print("2")
    max_chain_index = 0
    for pair in pairing:
        search_0 = np.searchsorted(all_ints_sorted, pair[0])
        search_1 = np.searchsorted(all_ints_sorted, pair[1])

        chain_ind_0 = chain_indices[search_0]
        chain_ind_1 = chain_indices[search_1]

        check_existing_chains_0 = (chain_ind_0 > 0)
        check_existing_chains_1 = (chain_ind_1 > 0)

        check_existing = (check_existing_chains_0 or check_existing_chains_1)
        if not check_existing:
            for point in pair:
                chain_indices[np.searchsorted(all_ints_sorted, point)] = max_chain_index + 1
            max_chain_index += 1
        else:
            if check_existing_chains_0 and (not check_existing_chains_1):
                chain_indices[search_1] = chain_ind_0
            elif check_existing_chains_1 and (not check_existing_chains_0):
                chain_indices[search_0] = chain_ind_1
    print("3")
    point_reduction = []
    for i in range(1, max_chain_index+1):
        point_reduction.append(all_ints_sorted[np.where(chain_indices == i)][::-1])
    final_send = np.array(point_reduction, dtype=object)
    if points_are_raw:
        for i in range(len(final_send)):
            row_vals, col_vals = np.floor_divide(final_send[i], max_final), np.mod(final_send[i], max_final)

            final_send[i] = np.swapaxes(np.array([row_vals, col_vals]), 0, 1)
            #final_send[i] = np.swapaxes(final_send[i], 0, 1)
    return final_send

def new_pool(bitmap):
    link_pairs_full = None
    shifts = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    for shift in shifts:
        shifted_bitmap = stego.shift(bitmap, shift, border_fill=-1)
        index_similar = np.where((bitmap == shifted_bitmap))
        index_reshifted = tuple([index_similar[0]-shift[0], index_similar[1]-shift[1]])
        linked_pairs = np.array([np.transpose(list(index_similar)), np.transpose(list(index_reshifted))])
        linked_pairs = np.swapaxes(linked_pairs, 0, 1)
        if link_pairs_full is None:
            link_pairs_full = linked_pairs
        else:
            link_pairs_full = np.concatenate((link_pairs_full, linked_pairs), axis=0)
    return link_pairs_full

def test_pairs():
    prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    pairs = []
    for prime in prime_numbers:
        for i in range(1, 7):
            pairs.append([prime**i, prime**(i+1)])
    pairs = np.array(pairs)
    print(pairs)
    reduced_pairs = reduce_pairing_order_v4(pairs)
    print(reduced_pairs)

def ultimate_graph_disjoint_size_assignment(array_in, reduce_iterations=None):
    pairs_get = new_pool(array_in)
    print("passing pairs")
    print(pairs_get)
    pairing_chains_get = reduce_pairing_order_v7(pairs_get, iterations=reduce_iterations)
    #pairing_chains_get = reduce_pairing_backend_v2(pairs_get)
    size_map = np.ones_like(array_in)
    sum_chains = 0
    for chain in pairing_chains_get:
        #print(chain[:, 0])
        #print(chain[:, 1])
        size_map[chain[:, 0], chain[:, 1]] = len(chain)
        sum_chains += 1
    print(sum_chains, "total chains")
    return size_map

def isolate_bit_image(img, bit, return_rgb_image=None):
    if return_rgb_image is None:
        return_rgb_image = False
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    div_value = 2**(7-bit)
    r, g, b = np.floor_divide(r, div_value), np.floor_divide(g, div_value), np.floor_divide(b, div_value)
    r, g, b = np.mod(r, 2), np.mod(g, 2), np.mod(b, 2)
    if return_rgb_image:
        return_set = np.zeros_like(img).astype('float')
        return_set[:, :, 0] = r
        return_set[:, :, 1] = g
        return_set[:, :, 2] = b
    else:
        return_set = np.add(np.multiply(4, r), np.add(np.multiply(2, g), b))
    return return_set

def main():

    test_greymap = [[0, 1, 1, 0],
                    [0, 1, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0]]
    test_greymap = np.array(test_greymap)

    test_map_2 = np.random.randint(0, 4, (13, 13))

    img = cv2.imread("../processed_images/morey_ecc_file_test.png")
    greymap = stego.get_greymap(img, 0)

    if False:
        time_old_method = time.time()
        pool_map, scrub_map = stego.pool_method_v3(greymap, threshold=50, gap=5)
        time_old_method = time.time()-time_old_method
        print("Old time:", time_old_method)

        cv2.imshow("greymap", greymap.astype('float'))
        cv2.waitKey(0)

    time_taken = time.time()
    pairs_unique = new_pool(greymap)
    #pairs_unique = new_pool(test_greymap)
    #pairs_unique = new_pool(test_map_2)
    time_taken = time.time() - time_taken
    time_taken = time.time()
    #reduced_pairs = reduce_pairing_order_v4(pairs_unique)
    reduced_pairs = reduce_pairing_order_v7(pairs_unique, iterations=100)
    #reduced_pairs = reduce_pairing_backend_v2(pairs_unique)

    time_taken = time.time() - time_taken

    bitmap_visual = np.zeros_like(test_map_2)

    sum_count = 0
    for chain in reduced_pairs:
        #print(chain)
        sum_count += len(chain)
        #bitmap_visual[chain[:,0], chain[:,1]] = len(chain)

    #print("Map:")
    #print(test_map_2)
    #print("Group")
    #print(bitmap_visual)

    #cv2.imshow("Points Referenced", bitmap_visual.astype('float'))
    #cv2.waitKey(0)
    print("total points referenced:", sum_count)


    #print("reduced pairs shape:", np.shape(reduced_pairs))
    #print("reduced_pairs\n", reduced_pairs)
    print("time taken:", time_taken)

def main_2():
    img = cv2.imread("C:/Users/subje/Downloads/JPow.jpg")
    #img = cv2.imread("C:/Users/subje/Downloads/dhop.jpg")
    lsb_layer = isolate_bit_image(img, 7)
    mask_size = ultimate_graph_disjoint_size_assignment(lsb_layer, reduce_iterations=10)
    float_mask = np.divide(mask_size.astype('float'), np.max(mask_size))
    cv2.imshow("Raw layer", np.multiply(isolate_bit_image(img, 7, return_rgb_image=True), 255).astype('uint8'))
    cv2.imshow("size_mask", float_mask)
    cv2.waitKey(0)
    cv2.imwrite("pool_mask.png", np.multiply(float_mask,255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite("lsb_mask.png", np.multiply(isolate_bit_image(img, 7, return_rgb_image=True), 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

def main_3():
    test_pair_set = [[0, 1], [1, 5], [5, 10], [2, 4], [4, 7], [7, 9], [3, 8], [6, 8]]
    test_pair_set = np.array(test_pair_set)
    #chains_make = reduce_pairing_order_v7(test_pair_set, iterations=None)
    chains_make = reduce_pairing_backend_v3(test_pair_set)

def main_4():
    test_1 = np.transpose([np.arange(500), np.add(np.arange(500), 1)])
    test_1[-1,1] = test_1[-1,0]
    break_flag = False
    it_count = 0
    while (not break_flag):
        old_hash = hash(test_1.tobytes())
        print("test_array iteration",it_count)
        print(test_1[0:50,:])
        test_1[:,1] = test_1[test_1[:,1],1]
        it_count += 1
        new_hash = hash(test_1.tobytes())
        if (old_hash == new_hash):
            break_flag = True
        

#main()
main_2()
#main_3()
#main_4()
#test_pairs()