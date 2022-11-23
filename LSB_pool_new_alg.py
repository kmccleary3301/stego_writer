import stego
import cv2
import time
import numpy as np
import bisect
import sys

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

def reduce_pairing_backend(pairing, pairings_are_sorted=None, pairings_are_unique=None, return_chains=None):
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

def reduce_pairing_order(pairing, iterations=None, pairings_are_sorted=None, pairings_are_unique=None):
    points_are_tuples = False
    if len(list(np.shape(pairing))) >= 3:
        points_are_tuples = True
        points_row, points_col = pairing[:, :, 0], pairing[:, :, 1]
        max_final = max(np.max(points_row) + 1, np.max(points_col) + 1)
        pairing = np.add(np.multiply(points_row, max_final), points_col).astype('int64')

    reduced_pairing = reduce_pairing_backend(pairing, pairings_are_sorted=pairings_are_sorted,
                                                pairings_are_unique=pairings_are_unique,
                                                return_chains=True)

    if not points_are_tuples:
        return reduced_pairing

    for i, chain in enumerate(reduced_pairing):
        reduced_pairing[i] = np.transpose([np.floor_divide(chain, max_final), np.mod(chain, max_final)])

    return np.array(reduced_pairing, dtype=object)

def new_pool(bitmap):
    bitmap = bitmap.astype('int64')
    print("bitmap shape ->",np.shape(bitmap))
    link_pairs_full = None
    shifts = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=np.inf)
    print("bitmap")
    print(bitmap)
    print("bitmap 17, 16 ->",bitmap[17,16])
    for shift in shifts:
        shifted_bitmap = stego.shift(bitmap, shift, border_fill=-1)
        index_similar = np.where((bitmap == shifted_bitmap))
        #print("shifted bitmap")
        #print(shifted_bitmap)
        print("shift",shift,"bitmap[17,16] ->",shifted_bitmap[17,16])
        print("vs bitmap[17,16] ->",bitmap[17,16])

        index_reshifted = tuple([index_similar[0]-shift[0], index_similar[1]-shift[1]])
        linked_pairs = np.array([np.transpose(list(index_similar)), np.transpose(list(index_reshifted))])
        linked_pairs = np.swapaxes(linked_pairs, 0, 1)
        #print("linked_pairs")
        #print(linked_pairs)

        find_target = np.where((linked_pairs[:,0,0] == 17) & (linked_pairs[:,0,1] == 15) & (linked_pairs[:,1,0] == 17) & (linked_pairs[:,1,1] == 16))
        if (find_target[0].size != 0):
            print("FOUND IT")
            print(find_target)
            print("shift ->",shift)

        if link_pairs_full is None:
            link_pairs_full = linked_pairs
        else:
            link_pairs_full = np.concatenate((link_pairs_full, linked_pairs), axis=0)
    np.set_printoptions(threshold=20)
    return link_pairs_full

def ultimate_graph_disjoint_size_assignment(array_in, reduce_iterations=None):
    pairs_get = new_pool(array_in)
    print("pool")
    #np.set_printoptions(threshold=sys.maxsize)
    #print(pairs_get)
    pairing_chains_get = reduce_pairing_order(pairs_get, iterations=reduce_iterations)
    size_map = np.ones_like(array_in).astype('uint64')
    sum_chains = 0
    for chain in pairing_chains_get:
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

def convert_255(img_array):
    dtype = type(np.ndarray.flatten(img_array)[0])
    min, max = np.min(img_array), np.max(img_array)

    if "int" in str(dtype):
        for i in range(3, 9):
            if (max <= 2**i and max >= 2**(i-1)):
                return np.multiply(np.divide(img_array, 2**i).astype('float'), 255).astype('uint8')
    elif "float" in str(dtype):
        return np.multiply(img_array, 255).astype('uint8')
    
    return img_array

def main():
    #img = cv2.imread("C:/Users/subje/Downloads/JPow.jpg")
    img = cv2.imread("C:/Users/subje/Downloads/dhop_mini_seg.png")
    lsb_layer = isolate_bit_image(img, 7)
    test_1 = convert_255(lsb_layer)
    mask_size = ultimate_graph_disjoint_size_assignment(lsb_layer, reduce_iterations=10)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=np.inf)
    print("mask")
    print(mask_size)
    np.set_printoptions(threshold=70)



    float_mask = np.divide(mask_size.astype('float'), np.max(mask_size))

    new_mask = np.minimum(mask_size, 255).astype('uint8')
    cv2.imshow("Raw layer", np.multiply(isolate_bit_image(img, 7, return_rgb_image=True), 255).astype('uint8'))
    cv2.imshow("size_mask", float_mask)
    cv2.waitKey(0)
    cv2.imwrite("pool_mask.png", np.multiply(float_mask,255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite("lsb_mask.png", convert_255(isolate_bit_image(img, 7, return_rgb_image=True)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite("mask_counted.png", new_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

main()
