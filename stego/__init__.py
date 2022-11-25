import cv2
import numpy as np
import random
import codecs

def to_bits(inStr, encodingType='utf-8'):
    m0 = inStr.encode(encodingType)
    m15 = ''.join([format(i, '#010b').replace('0b', '') for i in m0])
    return m15

def from_bits(inStr, encodingType='utf-8', pLen=8):
    return bytes([int(inStr[i:i+pLen], 2) for i in range(0, len(inStr)-pLen+1, pLen)]).decode(encodingType)

def numpy_checkerboard(shape):
    x = np.zeros(shape, dtype=int)
    x[1::2, ::2] = 1
    x[::2, 1::2] = 1
    return x

def bits_to_image(image, bit_string, point_indeces, end_message_flag=None):
    if end_message_flag is None:
        end_message_flag = '11011111111111111111111111111111'

    total_points = len(point_indeces[0])
    message_bits = len(bit_string)
    flag_bits = len(end_message_flag)
    bits_available = total_points*3

    print('\n\nWriting bits to image')
    print('Message size ->', message_bits, 'bits')
    print('Flag size    ->', flag_bits, 'bits')
    print('Total size   ->', flag_bits+message_bits, 'bits')
    print('Availability ->', bits_available, 'bits')

    point_set_hash = hash(point_indeces.data.tobytes())

    print('Point Collection Hash ->', point_set_hash)

    if (message_bits+flag_bits) > bits_available:
        print('\nWARNING: Not enough pixels available')
        target_size = bits_available - flag_bits
        print('Truncating size:', bits_available, '->', target_size)
        bit_string = bit_string[:target_size]

    bit_string += end_message_flag
    break_flag = False
    for index in range(len(point_indeces[0])):
        point_target = [point_indeces[0][index], point_indeces[1][index]]
        for channel in range(3):
            current_bit = (3*index) + channel
            if current_bit >= len(bit_string):
                break_flag = True
                break
            image[point_target[0]][point_target[1]][channel] -= (image[point_target[0]][point_target[1]][channel] % 2)
            image[point_target[0]][point_target[1]][channel] += int(bit_string[current_bit])
        if break_flag:
            break
    return image

def bits_from_image(image, point_indeces, end_message_flag=None):
    if end_message_flag is None:
        end_message_flag = '11011111111111111111111111111111'
    search_backward_length = 18*len(end_message_flag)
    num_pixels_given = len(point_indeces[0])
    bit_string_read = ''
    for index in range(num_pixels_given):
        point_target = [point_indeces[0][index], point_indeces[1][index]]
        current_pixel_bits = np.mod(image[point_target[0]][point_target[1]], 2)
        for bit in current_pixel_bits:
            bit_string_read += str(bit)
        if index%8 == 0:
            if bit_string_read[-search_backward_length:].find(end_message_flag) != -1:
                return bit_string_read[:bit_string_read.index(end_message_flag)]
    try:
        return bit_string_read[:bit_string_read.index(end_message_flag)]
    except ValueError:
        print(bit_string_read)

def shuffle_seed(list_in, seed):
    list_shape = np.shape(list_in)
    np.random.seed(seed)
    if len(list_shape) != 1:
        new_list = np.ndarray.flatten(np.array(list_in))
    else:
        new_list = np.array(list_in)
    data_length = len(new_list)
    shuf_order = np.arange(data_length)
    np.random.shuffle(shuf_order)
    shuffled_flat = new_list[shuf_order]
    return np.reshape(shuffled_flat, list_shape)

def unshuffle_seed(list_in, seed):
    list_shape = np.shape(list_in)
    np.random.seed(seed)
    if len(list_shape) != 1:
        new_list = np.ndarray.flatten(np.array(list_in))
    else:
        new_list = np.array(list_in)
    data_length = len(new_list)

    shuf_order = np.arange(data_length)
    np.random.shuffle(shuf_order)

    unshuf_order = np.zeros_like(shuf_order)
    unshuf_order[shuf_order] = np.arange(data_length)

    unshuffled_data = new_list[unshuf_order.astype('uint64')]

    return np.reshape(unshuffled_data, list_shape)

def string_shuffle_seed(string_in, seed):
    str_shuffle_array = shuffle_seed(list(string_in), seed)
    str_return = ''
    for char in str_shuffle_array:
        str_return += char
    return str_return

def string_unshuffle_seed(string_in, seed):
    str_unshuffle_array = unshuffle_seed(list(string_in), seed)
    str_return = ''
    for char in str_unshuffle_array:
        str_return += char
    return str_return

def shift(array, shift_vals, border_fill=None):
    if border_fill is None:
        border_fill = 0
    if type(shift_vals) is tuple:
        shift_vals = list(shift_vals)

    array_shape_list = list(np.shape(array))

    for axis in range(len(shift_vals)):
        offset = shift_vals[axis]
        if offset == 0:
            continue
        new_array_step_1 = np.roll(array, offset, axis=axis)


        if offset > 0:
            rewrite_indices = [i for i in range(offset)]
        else:
            rewrite_indices = [(array_shape_list[axis]-1-i) for i in range(abs(offset))]


        if axis == 0:
            #zero_slice = np.zeros_like(new_array_step_1[0])
            zero_slice = np.full_like(new_array_step_1[0], border_fill)
            for index in rewrite_indices:
                new_array_step_1[index] = zero_slice
        else:
            new_array_step_1 = np.swapaxes(new_array_step_1, 0, axis)
            zero_slice = np.full_like(new_array_step_1[0], border_fill)
            for index in rewrite_indices:
                new_array_step_1[index] = zero_slice
            new_array_step_1 = np.swapaxes(new_array_step_1, axis, 0)
        array = new_array_step_1
    return array

def search_sorted(lys, val):
    first = 0
    last = len(lys) - 1
    index = -1
    while (first <= last) and (index == -1):
        mid = (first + last) // 2
        if lys[mid] == val:
            index = mid
        else:
            if val < lys[mid]:
                last = mid - 1
            else:
                first = mid + 1
    if lys[index] != val:
        return -1
    return index

def pool_method_v3(greymap, threshold=None, gap=None):
    if threshold is None:
        threshold = 60
    if gap is None:
        gap = 15

    lower_threshold = threshold - gap

    greymap_shape = np.shape(greymap)

    shift_map = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    temp_map = np.ndarray.flatten(greymap)
    flat_indeces = np.arange(len(temp_map))
    base_map_numerical = np.reshape(flat_indeces, greymap_shape)

    curmap_final_prior = np.zeros_like(base_map_numerical)

    iterations = threshold
    curmap = base_map_numerical

    for iter in range(iterations):
        curstep = curmap
        for shift_list in shift_map:
            curstep = np.maximum(curstep, shift(curstep, shift_list))
        curstep = np.multiply(curstep, greymap)
        curmap = curstep
        if iter == iterations - 2:
            curmap_final_prior = curmap

    curmap_flat = np.ndarray.flatten(curmap)

    diff_min = np.minimum(curmap, curmap_final_prior)
    diff_max = np.maximum(curmap, curmap_final_prior)

    min_max_indices = np.where(diff_max != diff_min)

    pair_listing = [diff_max[min_max_indices], diff_min[min_max_indices]]
    pair_listing = np.swapaxes(pair_listing, 0, 1)

    pairing_remove_dup = np.unique(pair_listing, axis=0)

    reduced_pair_listing = reduce_pairing_order_v4(pairing_remove_dup)
    (unique, counts) = np.unique(curmap_flat, return_counts=True)

    new_unique_thresh, new_counts_thresh = custom_pooling_unique_counts_reduction(reduced_pair_listing, unique[1:],
                                                                                  counts[1:], threshold)
    new_unique_gap, new_counts_gap = custom_pooling_unique_counts_reduction(reduced_pair_listing, unique[1:],
                                                                            counts[1:], lower_threshold)

    threshold_unique = new_unique_thresh[new_counts_thresh >= threshold]

    gap_unique = new_unique_gap[new_counts_gap >= lower_threshold]

    curmap_over_threshold_flat = np.in1d(curmap_flat, threshold_unique)
    curmap_over_threshold = np.reshape(curmap_over_threshold_flat, greymap_shape)

    curmap_over_gap_flat = np.in1d(curmap_flat, gap_unique)
    curmap_over_gap = np.reshape(curmap_over_gap_flat, greymap_shape)

    curmap_over_gap = np.multiply(curmap_over_gap, np.subtract(1, curmap_over_threshold))

    return curmap_over_threshold, curmap_over_gap

def reduce_pairing_order_v2(pairing):
    reduced_v1 = []
    for pair in pairing:
        check_existing = False
        for chain_ind in range(len(reduced_v1)):
            if pair[0] in reduced_v1[chain_ind] or pair[1] in reduced_v1[chain_ind]:
                reduced_v1[chain_ind] = np.append(reduced_v1[chain_ind], pair)
                check_existing = True
        if not check_existing:
            reduced_v1.append(pair)
    reduced_v1 = np.array(reduced_v1, dtype=object)
    for i in range(len(reduced_v1)):
        reduced_v1[i] = np.unique(reduced_v1[i])
        reduced_v1[i] = np.sort(reduced_v1[i])[::-1]
    return reduced_v1

def reduce_pairing_order_v4(pairing): #time complexity is O(n*log(n)), extremely fast
    all_ints_sorted = np.sort(np.unique(np.ndarray.flatten(np.array(pairing))))
    chain_indices = np.zeros_like(all_ints_sorted)
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
    point_reduction = []
    for i in range(1, max_chain_index+1):
        point_reduction.append(all_ints_sorted[np.where(chain_indices == i)][::-1])
    return np.array(point_reduction, dtype=object)

def custom_pooling_unique_counts_reduction(reduced_pairing, unique, counts, threshold):
    new_order = np.lexsort([counts, unique])
    unique = unique[new_order]
    counts = counts[new_order]
    for ch_ind, chain in enumerate(reduced_pairing):
        count_total = 0
        for val_ind, val in enumerate(reduced_pairing[ch_ind]):
            unique_index = search_sorted(unique, val)
            count_total += counts[unique_index]
        if count_total >= threshold:
            for val_ind, val in enumerate(reduced_pairing[ch_ind]):
                unique_index = search_sorted(unique, val)
                if unique_index != -1:
                    counts[unique_index] = threshold
    return unique, counts

def expand_bitmap_white_area(bitmap_in, iterations):
    shift_map = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for iter in range(iterations):
        curstep = bitmap_in
        for shift_list in shift_map:
            curstep = np.maximum(curstep, shift(curstep, shift_list))
        bitmap_in = curstep
    return bitmap_in

def get_greymap(img, bit_value):
    bit_value = int(bit_value) % 8

    LSB_collection = np.mod(img, 2)

    LSP_list_shape = list(np.shape(img))

    new_shape = tuple(LSP_list_shape[:2])

    LSB_split = np.array_split(LSB_collection, 3, axis=2)

    exp_set = [2 ** i for i in range(len(LSB_split))]

    LSB_final = np.zeros(new_shape)

    for i in range(len(LSB_split)):
        LSB_final = np.add(LSB_final, np.multiply(np.reshape(LSB_split[i], new_shape), exp_set[i]))

    return np.equal(LSB_final, bit_value).astype('uint8')

def full_image_blob_map(img, threshold=50, gap=5):
    LSB_collection = np.mod(img, 2)

    LSP_list_shape = list(np.shape(img))

    new_shape = tuple(LSP_list_shape[:2])

    LSB_split = np.array_split(LSB_collection, 3, axis=2)

    exp_set = [2 ** i for i in range(len(LSB_split))]

    LSB_final = np.zeros(new_shape)

    for i in range(len(LSB_split)):
        LSB_final = np.add(LSB_final, np.multiply(np.reshape(LSB_split[i], new_shape), exp_set[i]))

    LSB_greymaps = []

    for bit in range(2 ** (len(LSB_split))):
        temp_array = np.equal(LSB_final, bit).astype('uint8')
        LSB_greymaps.append(temp_array)

    pool_data = np.zeros_like(LSB_greymaps[0])
    scrub_data = pool_data

    i, count = 1, len(LSB_greymaps)

    for map in LSB_greymaps:
        print('Processing bitmap', i, 'out of', count)
        #pool_map, scrub_map = pool_greyscale_island_size_old(map, threshold=threshold, gap=gap, over_factor=8)
        pool_map, scrub_map = pool_method_v3(map, threshold=threshold, gap=gap)
        pool_data = np.maximum(pool_map, pool_data)
        scrub_data = np.maximum(scrub_map, scrub_data)
        i += 1

    scrub_data = np.multiply(np.subtract(1, pool_data), scrub_data)

    return pool_data, scrub_data

def sanitize_image(image, target_bitmap):
    index_set = np.where(target_bitmap==1)
    ht, wd = len(image), len(image[0])
    write_template = [[i // 4, ((i // 2) % 2), i % 2] for i in range(8)]
    for i in range(len(index_set[0])):
        point = [index_set[0][i], index_set[1][i]]
        cur_template = [[ind // 4, ((ind // 2) % 2), ind % 2] for ind in range(8)]
        for g in [[point[0] + k[0], point[1] + k[1]] for k in [[-1, 0], [1, 0], [0, -1], [0, 1]]]:
            if 0 <= g[0] < ht and 0 <= g[1] < wd:
                pix_bits = np.mod(image[g[0]][g[1]], 2)
                color_bit = sum([pix_bits[in_2]*(2**(2-in_2)) for in_2 in range(3)])
                if write_template[color_bit] in cur_template:
                    cur_template.remove(write_template[color_bit])
        new_color = cur_template[random.randrange(0, len(cur_template))]
        for q in range(3):
            image[point[0]][point[1]][q] -= (image[point[0]][point[1]][q] % 2)
            image[point[0]][point[1]][q] += new_color[q]
    return image

def image_lsb_display(image):
    LSB_collection = np.mod(image, 2)
    return np.multiply(LSB_collection, 255)

def convert_file_to_string_data(subject_file_path, subject_file_encoding=None):
    if subject_file_encoding is None:
        subject_file_encoding = "ansi"

    final_string = ""

    BLOCKSIZE = 1048576  # or some other, desired size in bytes
    with codecs.open(subject_file_path, "r", subject_file_encoding) as sourceFile:
        file_name = subject_file_path
        if '/' in file_name:
            file_name = file_name[len(file_name) - file_name[::-1].index('/'):]
        final_string += file_name + "\n"
        final_string += subject_file_encoding + "\n"
        while True:
            contents = sourceFile.read(BLOCKSIZE)
            if not contents:
                break
            final_string += contents
    return final_string

def convert_string_to_file_data(file_string):
    file_name_found, encoding_found = False, False

    if '\n' in file_string:
        break_index = file_string.index('\n')
        file_name = file_string[:break_index]
        file_string = file_string[break_index + 1:]
        file_name_found = True
    if '\n' in file_string:
        break_index = file_string.index('\n')
        encoding_type = file_string[:break_index]
        file_string = file_string[break_index + 1:]
        encoding_found = True
    if not (file_name_found and encoding_found):
        return file_string, "", ""
    return file_string, file_name, encoding_type

def write_file_from_tuple_data(file_name, file_content, encoding_type, path_to=None):
    if path_to is None:
        path_to = ""
    else:
        if path_to[-1] != '/':
            path_to += '/'
    file_path = path_to + file_name
    #BLOCKSIZE = 1048576
    with codecs.open(file_path, "w", encoding_type) as targetFile:
        targetFile.write(file_content)

#new methods
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

def reduce_pairing_backend(pairing, pairings_are_sorted=None, pairings_are_unique=None, return_chains=None, bar_values=None):
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
    if bar_values is None:
        bar_values = {
            "value": 0,
            "p_bar_label": "Encoding",
            "step_integer": 0
        }
    all_ints_sorted = faster_unique_2d(np.ndarray.flatten(pairing))
    bar_values["step_integer"] += 1
    reference_array = np.swapaxes([np.arange(len(all_ints_sorted)), np.ones_like(all_ints_sorted)], 0, 1)
    pairing_dims_get = np.shape(pairing)
    new_pairs = np.ndarray.flatten(pairing)
    bar_values["step_integer"] += 1

    print("pairing_flattened")

    new_pairs = np.searchsorted(all_ints_sorted, new_pairs)
    new_pairs = np.reshape(new_pairs, pairing_dims_get)

    loop_flag = True
    iteration_count = 0
    active_pairs = new_pairs

    assignments_1 = reference_array[active_pairs[:,0],0]
    assignments_2 = reference_array[active_pairs[:,1],0]
    bar_values["step_integer"] += 1
    
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
        bar_values["step_integer"] += 0.5**(iteration_count)
    
    bar_values["step_integer"] = int(np.ceil(bar_values["step_integer"]))
    

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
    bar_values["step_integer"] += 1

    chain_markers = np.not_equal(groups_list, np.concatenate((groups_list[1:], [groups_list[-1] + 1])))
    chain_markers = np.add(np.where(chain_markers), 1)
    chain_markers = np.concatenate(([0], chain_markers[0]))

    bar_values["step_integer"] += 1
    chains_return = []

    for i in range(len(chain_markers) - 1):
        temp_chain = point_group_list[chain_markers[i]:chain_markers[i + 1], 1]
        chains_return.append(temp_chain)
    bar_values["step_integer"] += 1
    return np.array(chains_return, dtype=object)

def reduce_pairing_order(pairing, iterations=None, pairings_are_sorted=None, pairings_are_unique=None, bar_values=None):
    if bar_values is None:
        bar_values = {
            "value": 0,
            "p_bar_label": "Encoding",
            "step_integer": 0
        }
    points_are_tuples = False
    if len(list(np.shape(pairing))) >= 3:
        points_are_tuples = True
        points_row, points_col = pairing[:, :, 0], pairing[:, :, 1]
        max_final = max(np.max(points_row) + 1, np.max(points_col) + 1)
        pairing = np.add(np.multiply(points_row, max_final), points_col).astype('int64')

    reduced_pairing = reduce_pairing_backend(pairing, pairings_are_sorted=pairings_are_sorted,
                                                pairings_are_unique=pairings_are_unique,
                                                return_chains=True, bar_values=bar_values)

    if not points_are_tuples:
        return reduced_pairing

    for i, chain in enumerate(reduced_pairing):
        reduced_pairing[i] = np.transpose([np.floor_divide(chain, max_final), np.mod(chain, max_final)])

    return np.array(reduced_pairing, dtype=object)

def new_pool(bitmap, search_area=None):
    if search_area is None:
        search_area = np.ones_like(bitmap)
    print("bitmap shape ->",np.shape(bitmap))
    link_pairs_full = None
    shifts = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    for shift_temp in shifts:
        shifted_bitmap = shift(bitmap, shift_temp, border_fill=-1)
        index_similar = np.where((bitmap == shifted_bitmap) & (search_area == 1))
        index_reshifted = tuple([index_similar[0]-shift_temp[0], index_similar[1]-shift_temp[1]])
        linked_pairs = np.array([np.transpose(list(index_similar)), np.transpose(list(index_reshifted))])
        linked_pairs = np.swapaxes(linked_pairs, 0, 1)
        if link_pairs_full is None:
            link_pairs_full = linked_pairs
        else:
            link_pairs_full = np.concatenate((link_pairs_full, linked_pairs), axis=0)
    return link_pairs_full

def image_size_assignment(array_in, reduce_iterations=None, search_area=None, bar_values=None):
    if bar_values is None:
        bar_values = {
            "value": 0,
            "p_bar_label": "Encoding",
            "step_integer": 0
        }
    pairs_get = new_pool(array_in, search_area=search_area)
    pairing_chains_get = reduce_pairing_order(pairs_get, iterations=reduce_iterations, bar_values=bar_values)
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

def pool_mask_visual(array_in, is_size_assignment=None):
    if is_size_assignment is None:
        is_size_assignment = False
    if is_size_assignment:
        float_mask = np.divide(np.sqrt(array_in).astype('float'), np.sqrt(np.max(array_in)))
    else:
        size_assignment = image_size_assignment(array_in)
        float_mask = np.divide(np.sqrt(size_assignment).astype('float'), np.sqrt(np.max(size_assignment)))
    return np.multiply(float_mask, 255).astype('uint8')

def gen_square_diamond(size):
    grid = np.zeros((size*2+1, size*2+1))
    grid_shape = np.shape(grid)
    grid_center = [(grid_shape[0] // 2), (grid_shape[1] // 2)]
    grid[grid_center[0], grid_center[1]] = 1
    
    for radius in range(size+1):
       
        for dir in range(4):
            origin = np.copy(grid_center)
            origin[dir%2] += radius*(-1)**(dir // 2)
            for iteration in range(radius):
                coord = np.copy(origin)
                coord[0] += iteration*((-1)**(dir // 2 + 1))
                coord[1] += iteration*((-1)**((dir + 3) // 2 + 1))
                grid[coord[0], coord[1]] = 1
    calculated_size = (2*size+1)**2 - 2*(size+1)*(size)
    return grid

def tile_diamond(shape, diamond):
       
    diamond_shape = list(np.shape(diamond))

    rows_to_add = diamond_shape[0]-diamond_shape[1]
    t_grid = np.zeros((10*diamond_shape[0], 10*diamond_shape[1])) 

    half_increments = [diamond_shape[0]-diamond_shape[1]//2, diamond_shape[1]//2 + 1]
    full_increments = np.copy(diamond_shape)
    full_increments[0] += diamond_shape[0]-diamond_shape[1]+1
    full_increments[1] += 1
    grid = np.zeros(shape)

    for y in range(1,5):
        for x in range(1,5):
            c_1 = y*full_increments[0]
            r_1 = x*full_increments[1]
            c_2 = c_1+half_increments[0]
            r_2 = r_1+half_increments[1]
            t_grid[c_1:c_1+diamond_shape[0],r_1:r_1+diamond_shape[1]] = np.maximum(t_grid[c_1:c_1+diamond_shape[0],r_1:r_1+diamond_shape[1]], diamond)
            t_grid[c_2:c_2+diamond_shape[0],r_2:r_2+diamond_shape[1]] = np.maximum(t_grid[c_2:c_2+diamond_shape[0],r_2:r_2+diamond_shape[1]], diamond)

    rect_start_col = 2*full_increments[0]
    rect_start_row = 2*full_increments[1]
    rect_size_col = diamond_shape[0]+rows_to_add+1
    rect_size_row = diamond_shape[1]+1
    rect_end_col = rect_start_col+2*rect_size_col
    rect_end_row = rect_start_row+2*rect_size_row
    
    grid_template = np.copy(t_grid[rect_start_col:rect_end_col,rect_start_row:rect_end_row])
    shape_multiples = list(np.shape(grid_template))

    temp_shape_adjust = list(shape)
    temp_shape_adjust[0] += 3*shape_multiples[0] - (temp_shape_adjust[0]%shape_multiples[0])
    temp_shape_adjust[1] += 3*shape_multiples[1] - (temp_shape_adjust[1]%shape_multiples[1])

    grid = np.zeros(tuple(temp_shape_adjust))
    for col in range(temp_shape_adjust[0] // shape_multiples[0]):
        for row in range(temp_shape_adjust[1] // shape_multiples[1]):
            col_pos = col*shape_multiples[0]
            row_pos = row*shape_multiples[1]
            grid[col_pos:col_pos+shape_multiples[0],row_pos:row_pos+shape_multiples[1]] = np.maximum(grid[col_pos:col_pos+shape_multiples[0],row_pos:row_pos+shape_multiples[1]], grid_template)
    return grid[0:shape[0],0:shape[1]]

def gen_exclusion_pattern(shape, size):
    if size <= 4: #just a checkerboard
        x = np.zeros(shape, dtype=int)
        x[1::2, ::2] = 1
        x[::2, 1::2] = 1
        return x
    square_radius = np.floor( np.sqrt((size-0.5)/2) - 0.5).astype('uint32')
    first_diamond_size = int((2*square_radius+1)**2 - 2*(square_radius+1)*(square_radius))
    square_diamond_get = gen_square_diamond(square_radius)
    added_row_size = np.shape(square_diamond_get)[1]
    remaining_size_capacity = size - first_diamond_size
    rows_to_add = np.maximum(0, remaining_size_capacity // added_row_size)
    row_add_template = np.tile([1], (added_row_size))
    row_insert_indices = np.arange(rows_to_add)+(np.shape(square_diamond_get)[0]//2)
    diamond_final = np.insert(square_diamond_get, row_insert_indices, row_add_template, axis=0)
    return tile_diamond(shape,diamond_final)

#/end new methods

def image_write_processing(image, string_input, base_key=None, string_shuffle_key=None,
                           custom_key_shuffle_key=None, random_key_pixels_assigned=None,
                           initial_points_shuffle_key=None, threshold=None, gap=None):
    key_max = 4294967295

    if base_key is None:
        base_key = 69298381

    if string_shuffle_key is None:
        string_shuffle_key = 58478232

    if custom_key_shuffle_key is None:
        custom_key_shuffle_key = 8274810

    if initial_points_shuffle_key is None:
        initial_points_shuffle_key = 98592859

    if random_key_pixels_assigned is None:
        random_key_pixels_assigned = 400

    base_key %= key_max
    string_shuffle_key %= key_max
    custom_key_shuffle_key %= key_max
    initial_points_shuffle_key %= key_max


    custom_key = random.randrange(0, 4294967293)

    image_shape = np.shape(image)

    image_shape = image_shape[:2]

    pool_test, scrub_test = full_image_blob_map(image, threshold=threshold, gap=gap)

    blob_test = expand_bitmap_white_area(pool_test, 5)

    target_data = np.multiply(scrub_test, np.subtract(1, blob_test))

    image_sanitized = sanitize_image(image, target_data)

    available_data = np.subtract(1, blob_test)

    available_data_expand = expand_bitmap_white_area(available_data, 1)

    checkerboard_mask = numpy_checkerboard(image_shape)

    checkerboard_mask_inv = np.subtract(1, checkerboard_mask)

    available_data_v1 = np.multiply(checkerboard_mask, available_data)

    scrub_down_target = np.multiply(checkerboard_mask_inv, available_data_expand)

    available_data_point_set = np.where(available_data_v1 == 1)

    point_set_step_2 = np.zeros_like(available_data_point_set)
    for i in range(2):
        point_set_step_2[i] = shuffle_seed(available_data_point_set[i], initial_points_shuffle_key)

    base_key_points = [point_set_step_2[i][:random_key_pixels_assigned] for i in range(2)]

    message_points = [point_set_step_2[i][random_key_pixels_assigned:] for i in range(2)]

    base_key_points_shuffled = np.zeros_like(base_key_points)

    for i in range(2):
        base_key_points_shuffled[i] = shuffle_seed(base_key_points[i], base_key)

    custom_key_string_bits = to_bits(str(custom_key))

    custom_key_bits_shuffle = string_shuffle_seed(custom_key_string_bits, custom_key_shuffle_key)

    img_step_1 = bits_to_image(image_sanitized, custom_key_bits_shuffle, base_key_points_shuffled)

    message_points_shuffled = np.zeros_like(message_points)

    for i in range(2):
        message_points_shuffled[i] = shuffle_seed(message_points[i], custom_key)

    string_bits = to_bits(string_input)

    string_bits_shuffle = string_shuffle_seed(string_bits, string_shuffle_key)

    img_step_2 = bits_to_image(img_step_1, string_bits_shuffle, message_points_shuffled)

    #final_image_step_2 = sanitize_image(img_step_2, scrub_down_target)
    #return final_image_step_2

    return img_step_2

def image_read_processing(image, base_key=None, string_shuffle_key=None,
                           custom_key_shuffle_key=None, random_key_pixels_assigned=None,
                          initial_points_shuffle_key=None, threshold=None, gap=None):
    key_max = 4294967295

    if base_key is None:
        base_key = 69298381

    if string_shuffle_key is None:
        string_shuffle_key = 58478232

    if custom_key_shuffle_key is None:
        custom_key_shuffle_key = 8274810

    if initial_points_shuffle_key is None:
        initial_points_shuffle_key = 98592859

    if random_key_pixels_assigned is None:
        random_key_pixels_assigned = 400

    base_key %= key_max
    string_shuffle_key %= key_max
    custom_key_shuffle_key %= key_max
    initial_points_shuffle_key %= key_max

    image_shape = np.shape(image)

    image_shape = image_shape[:2]

    pool_test, scrub_test = full_image_blob_map(image, threshold=threshold, gap=gap)

    blob_test = expand_bitmap_white_area(pool_test, 5)

    available_data = np.subtract(1, blob_test)

    checkerboard_mask = numpy_checkerboard(image_shape)

    available_data_v1 = np.multiply(checkerboard_mask, available_data)

    available_data_point_set = np.where(available_data_v1 == 1)

    point_set_step_2 = np.zeros_like(available_data_point_set)
    for i in range(2):
        point_set_step_2[i] = shuffle_seed(available_data_point_set[i], initial_points_shuffle_key)

    base_key_points = [point_set_step_2[i][:random_key_pixels_assigned] for i in range(2)]

    message_points = [point_set_step_2[i][random_key_pixels_assigned:] for i in range(2)]

    base_key_points_shuffled = np.zeros_like(base_key_points)

    for i in range(2):
        base_key_points_shuffled[i] = shuffle_seed(base_key_points[i], base_key)

    custom_key_string_bits_shuffled = bits_from_image(image, base_key_points_shuffled)

    custom_key_string_bits = string_unshuffle_seed(custom_key_string_bits_shuffled, custom_key_shuffle_key)

    custom_key = int(from_bits(custom_key_string_bits))

    message_points_shuffled = np.zeros_like(message_points)

    for i in range(2):
        message_points_shuffled[i] = shuffle_seed(message_points[i], custom_key)

    string_bits_shuffled = bits_from_image(image, message_points_shuffled) #good to here

    string_bits = string_unshuffle_seed(string_bits_shuffled, string_shuffle_key)

    message_string = from_bits(string_bits)

    return message_string

def image_write_check_readable(image, string_input, base_key=None, string_shuffle_key=None,
                           custom_key_shuffle_key=None, random_key_pixels_assigned=None,
                           initial_points_shuffle_key=None, threshold=None, gap=None,
                               sanitize_second_step=None):
    key_max = 4294967295

    if base_key is None:
        base_key = 69298381

    if string_shuffle_key is None:
        string_shuffle_key = 58478232

    if custom_key_shuffle_key is None:
        custom_key_shuffle_key = 8274810

    if initial_points_shuffle_key is None:
        initial_points_shuffle_key = 98592859

    if random_key_pixels_assigned is None:
        random_key_pixels_assigned = 400

    if sanitize_second_step is None:
        sanitize_second_step = False

    base_key %= key_max
    string_shuffle_key %= key_max
    custom_key_shuffle_key %= key_max
    initial_points_shuffle_key %= key_max

    custom_key = random.randrange(0, 4294967293)

    image_shape = np.shape(image)

    image_shape = image_shape[:2]

    pool_test, scrub_test = full_image_blob_map(image, threshold=threshold, gap=gap)

    pool_hash_initial = hash(pool_test.data.tobytes())

    blob_test = expand_bitmap_white_area(pool_test, 5)

    target_data = np.multiply(scrub_test, np.subtract(1, blob_test))

    image_sanitized = sanitize_image(image, target_data)

    available_data = np.subtract(1, blob_test)

    available_data_expand = expand_bitmap_white_area(available_data, 1)

    checkerboard_mask = numpy_checkerboard(image_shape)

    checkerboard_mask_inv = np.subtract(1, checkerboard_mask)

    available_data_v1 = np.multiply(checkerboard_mask, available_data)

    scrub_down_target = np.multiply(checkerboard_mask_inv, available_data_expand)

    available_data_point_set = np.where(available_data_v1 == 1)

    point_set_step_2 = np.zeros_like(available_data_point_set)
    for i in range(2):
        point_set_step_2[i] = shuffle_seed(available_data_point_set[i], initial_points_shuffle_key)

    base_key_points = [point_set_step_2[i][:random_key_pixels_assigned] for i in range(2)]

    message_points = [point_set_step_2[i][random_key_pixels_assigned:] for i in range(2)]

    base_key_points_shuffled = np.zeros_like(base_key_points)

    for i in range(2):
        base_key_points_shuffled[i] = shuffle_seed(base_key_points[i], base_key)

    custom_key_string_bits = to_bits(str(custom_key))

    custom_key_bits_shuffle = string_shuffle_seed(custom_key_string_bits, custom_key_shuffle_key)

    img_step_1 = bits_to_image(image_sanitized, custom_key_bits_shuffle, base_key_points_shuffled)

    message_points_shuffled = np.zeros_like(message_points)

    for i in range(2):
        message_points_shuffled[i] = shuffle_seed(message_points[i], custom_key)

    string_bits = to_bits(string_input)

    string_bits_shuffle = string_shuffle_seed(string_bits, string_shuffle_key)

    img_step_2 = bits_to_image(img_step_1, string_bits_shuffle, message_points_shuffled)

    if sanitize_second_step:
        img_step_2 = sanitize_image(img_step_2, scrub_down_target)

    verification_pool, temp = full_image_blob_map(img_step_2, threshold=threshold, gap=gap)

    pool_hash_final = hash(verification_pool.data.tobytes())

    clean_count = 0

    while pool_hash_final != pool_hash_initial and clean_count < 3:
        print('Pooling mismatch encountered. Attempting cleanup %i' % (clean_count + 1))
        pool_different = np.multiply(verification_pool, np.subtract(1, pool_test))
        scrub_area = np.multiply(pool_different, checkerboard_mask_inv)
        img_step_2 = sanitize_image(img_step_2, scrub_area)
        verification_pool, temp = full_image_blob_map(img_step_2, threshold=threshold, gap=gap)
        pool_hash_final = hash(verification_pool.data.tobytes())
        clean_count += 1

    if pool_hash_final != pool_hash_initial:
        print('WARNING: POOLING IS ALTERED, MESSAGE WILL BE UNREADABLE')
        print('RETURNING INITIAL IMAGE')
        return image

    print('Message successfully written')
    return img_step_2

def image_write_new(img_in, message, shuffle_key=None, threshold=None, cover_flag=None, blob_expand_size=None, bar_values=None, size_map=None):
    img = np.copy(img_in)
    if threshold is None:
        threshold = 20
    if shuffle_key is None:
        shuffle_key = 17876418
    if cover_flag is None:
        cover_flag = False
    if blob_expand_size is None:
        blob_expand_size = 5
    if bar_values is None:
        bar_values = {
            "value": 0,
            "p_bar_label": "Encoding",
            "step_integer": 0
        }

    shuffle_key %= (2**32)-3
    recovery_key = np.sin(shuffle_key).astype('float64')
    recovery_key = recovery_key - np.floor(recovery_key)
    recovery_key = np.floor((10**9)*recovery_key)
    recovery_key = (recovery_key % ((2**32)-3)).astype('uint32')
    print("recovery_key ->",recovery_key)

    lsb_layer = isolate_bit_image(img, 7)
    
    bar_values["step_integer"] += 1
    if size_map is None:
        image_size_assigned = image_size_assignment(lsb_layer, bar_values=bar_values)
    else:
        image_size_assigned = size_map
        bar_values["step_integer"] += 8
    bar_values["step_integer"] += 1
    pool_test = np.zeros_like(lsb_layer)
    pool_test[np.where(image_size_assigned >= threshold)] = 1
    bar_values["step_integer"] += 1
    print("pool_hash_write ->",hash(pool_test.tobytes()))
    bar_values["step_integer"] += 1
    blob = expand_bitmap_white_area(pool_test, blob_expand_size)
    bar_values["step_integer"] += 1
    #scrub_area = blob*(1-pool_test)
    #new_image = sanitize_image(img, scrub_area)

    scrub_area_2 = np.zeros_like(lsb_layer)
    bar_values["step_integer"] += 1
    scrub_area_2[np.where(image_size_assigned >= threshold-5)] = 1
    bar_values["step_integer"] += 1
    scrub_area_2 = scrub_area_2*(1-expand_bitmap_white_area(pool_test, blob_expand_size-1))
    bar_values["step_integer"] += 1
    #scrub_area_2 = scrub_area_2*(1-blob)
    #scrub_area_2 = scrub_area_2*(1-blob)

    new_image = sanitize_image(img, scrub_area_2)
    bar_values["step_integer"] += 1
    writable_pattern = gen_exclusion_pattern(np.shape(lsb_layer), threshold-5)
    bar_values["step_integer"] += 1
    target_set = (1-blob)*writable_pattern
    bar_values["step_integer"] += 1

    print("total bits before pattern correction ->",np.sum((1-blob)))
    print("total bits after  pattern correction ->",np.sum(target_set))

    target_set = np.where(target_set == 1)
    target_set = np.array(list(target_set))

    for i in range(2):
        target_set[i] = shuffle_seed(target_set[i], shuffle_key)

    scrub_set = (1-blob)*(1-writable_pattern)
    bar_values["step_integer"] += 1
    bit_make = to_bits(message)
    bit_make = string_shuffle_seed(bit_make, recovery_key)
    new_image = bits_to_image(new_image, bit_make, target_set)
    bar_values["step_integer"] += 1

    #Both options below prevent pool corruption
    #However, the latter option makes it somewhat visible using the debuging methods shown here
    #The first option computes the pool size again. For large images, this time doubling will be noticable
    #Enabling cover flag makes the write much less detectable.

    if cover_flag: 
        new_size_area = (1-pool_test)
        size_get_step_3 = image_size_assigned
        new_sizes = image_size_assignment(isolate_bit_image(new_image, 7), search_area=new_size_area)
        size_get_step_3[np.where((new_size_area == 1))] = new_sizes[np.where((new_size_area == 1))]
        final_scrub_set = np.zeros_like(scrub_set)
        final_scrub_set[np.where(size_get_step_3 >= threshold)] = 1
        final_scrub_set = final_scrub_set*(scrub_set)
        new_image = sanitize_image(new_image, final_scrub_set)
    else:
        new_image = sanitize_image(new_image, scrub_set)
    bar_values["step_integer"] += 1
    return new_image

def image_read_new(img, shuffle_key=None, threshold=None, blob_expand_size=None):
    if threshold is None:
        threshold = 20
    if shuffle_key is None:
        shuffle_key = 17876418
    if blob_expand_size is None:
        blob_expand_size = 5


    shuffle_key %= (2**32)-3
    recovery_key = np.sin(shuffle_key).astype('float64')
    recovery_key = recovery_key - np.floor(recovery_key)
    recovery_key = np.floor((10**9)*recovery_key)
    recovery_key = (recovery_key % ((2**32)-3)).astype('uint32')
    print("recovery_key ->",recovery_key)

    lsb_layer = isolate_bit_image(img, 7)
    image_size_assigned = image_size_assignment(lsb_layer)

    float_mask = np.divide(np.sqrt(image_size_assigned).astype('float'), np.sqrt(np.max(image_size_assigned)))
    cv2.imwrite("mask_generated_1.png", np.multiply(float_mask,255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

    print("image_size_assigned")
    print(image_size_assigned)

    pool_test = np.zeros_like(lsb_layer)
    pool_test[np.where(image_size_assigned >= threshold)] = 1

    print("pool_hash_read ->",hash(pool_test.tobytes()))

    blob = expand_bitmap_white_area(pool_test, blob_expand_size)

    writable_pattern = gen_exclusion_pattern(np.shape(lsb_layer), threshold-5)

    target_set = (1-blob)*writable_pattern

    print("total bits before pattern correction ->",np.sum((1-blob)))
    print("total bits after  pattern correction ->",np.sum(target_set))

    target_set = np.where(target_set == 1)
    target_set = np.array(list(target_set))
    for i in range(2):
        target_set[i] = shuffle_seed(target_set[i], shuffle_key)

    bits_read = bits_from_image(img, target_set)
    bits_read = string_unshuffle_seed(bits_read, recovery_key)

    message_read = from_bits(bits_read)

    return message_read


