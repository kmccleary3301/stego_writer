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



