import cv2
import numpy as np
import random


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

def shift(array, shift_vals):
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
            zero_slice = np.zeros_like(new_array_step_1[0])
            for index in rewrite_indices:
                new_array_step_1[index] = zero_slice
        else:
            new_array_step_1 = np.swapaxes(new_array_step_1, 0, axis)
            zero_slice = np.zeros_like(new_array_step_1[0])
            for index in rewrite_indices:
                new_array_step_1[index] = zero_slice
            new_array_step_1 = np.swapaxes(new_array_step_1, axis, 0)
        array = new_array_step_1
    return array

def pool_greyscale_island_size(greymap, threshold=50, gap=5):
    lower_bound = threshold-gap

    greymap_shape = np.shape(greymap)

    shift_map = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    temp_map = np.ndarray.flatten(greymap)
    flat_indeces = np.arange(len(temp_map))
    base_map_numerical = np.reshape(flat_indeces, greymap_shape)

    iterations = threshold
    curmap = base_map_numerical
    for iter in range(iterations):
        curstep = curmap
        for shift_list in shift_map:
            curstep = np.maximum(curstep, shift(curstep, shift_list))
        curstep = np.multiply(curstep, greymap)
        curmap = curstep
    curmap = np.multiply(curmap, greymap)

    curmap_flat = np.ndarray.flatten(curmap)

    (unique, counts) = np.unique(curmap_flat, return_counts=True)

    step_1 = unique[counts >= threshold]

    step_1_gap = unique[counts >= lower_bound]

    step_2 = np.in1d(curmap_flat, step_1)

    step_2_gap = np.in1d(curmap_flat, step_1_gap)

    step_2_reshape = np.reshape(step_2.astype('uint8'), greymap_shape)

    step_2_reshape = np.multiply(step_2_reshape, greymap)

    step_3 = step_2_reshape

    for i in range(250):
        for shift_list in shift_map:
            step_3_roll = shift(step_3, shift_list)
            step_3 = np.maximum(step_3, np.multiply(step_3_roll, greymap))



    step_2_gap_reshape = np.reshape(step_2_gap.astype('uint8'), greymap_shape)

    step_2_gap_reshape = np.multiply(np.subtract(1, step_2_reshape), step_2_gap_reshape)

    step_2_gap_reshape = np.multiply(step_2_gap_reshape, greymap)

    step_3_gap = step_2_gap_reshape

    for i in range(250):
        for shift_list in shift_map:
            step_3_gap_roll = shift(step_3_gap, shift_list)
            step_3_gap = np.maximum(step_3_gap, np.multiply(step_3_gap_roll, greymap))

    step_3_gap = np.multiply(np.subtract(1, step_3), step_3_gap)

    return step_3, step_3_gap

def pool_greyscale_island_size_old(greymap, threshold=50, gap=5, over_factor=5):
    lower_bound = threshold - gap

    greymap_shape = np.shape(greymap)

    shift_map = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    temp_map = np.ndarray.flatten(greymap)
    flat_indeces = np.arange(len(temp_map))
    base_map_numerical = np.reshape(flat_indeces, greymap_shape)

    iterations = threshold*over_factor
    curmap = base_map_numerical
    for iter in range(iterations):
        curstep = curmap
        for shift_list in shift_map:
            curstep = np.maximum(curstep, shift(curstep, shift_list))
        curstep = np.multiply(curstep, greymap)
        curmap = curstep
    curmap = np.multiply(curmap, greymap)

    curmap_flat = np.ndarray.flatten(curmap)

    (unique, counts) = np.unique(curmap_flat, return_counts=True)

    step_1 = unique[counts >= threshold]

    step_1_gap = unique[counts >= lower_bound]

    step_2 = np.in1d(curmap_flat, step_1)

    step_2_gap = np.in1d(curmap_flat, step_1_gap)

    step_2_reshape = np.reshape(step_2.astype('uint8'), greymap_shape)

    step_2_reshape = np.multiply(step_2_reshape, greymap)

    step_2_gap_reshape = np.reshape(step_2_gap.astype('uint8'), greymap_shape)

    step_2_gap_reshape = np.multiply(np.subtract(1, step_2_reshape), step_2_gap_reshape)

    step_2_gap_reshape = np.multiply(step_2_gap_reshape, greymap)

    return step_2_reshape, step_2_gap_reshape

def expand_bitmap_white_area(bitmap_in, iterations):
    shift_map = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for iter in range(iterations):
        curstep = bitmap_in
        for shift_list in shift_map:
            curstep = np.maximum(curstep, shift(curstep, shift_list))
        bitmap_in = curstep
    return bitmap_in

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
        pool_map, scrub_map = pool_greyscale_island_size_old(map, threshold=threshold, gap=gap, over_factor=8)
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

def image_write_processing(image, string_input, base_key=None, string_shuffle_key=None,
                           custom_key_shuffle_key=None, random_key_pixels_assigned=None):
    key_max = 4294967295

    if base_key is None:
        base_key = 69298381

    if string_shuffle_key is None:
        string_shuffle_key = 58478232

    if custom_key_shuffle_key is None:
        custom_key_shuffle_key = 8274810

    if random_key_pixels_assigned is None:
        random_key_pixels_assigned = 400

    base_key %= key_max
    string_shuffle_key %= key_max
    custom_key_shuffle_key %= key_max


    custom_key = random.randrange(0, 4294967293)

    image_shape = np.shape(image)

    image_shape = image_shape[:2]

    pool_test, scrub_test = full_image_blob_map(image, threshold=70, gap=40)

    blob_test = expand_bitmap_white_area(pool_test, 5)

    target_data = np.multiply(scrub_test, np.subtract(1, blob_test))

    image_sanitized = sanitize_image(image, target_data)

    available_data = np.subtract(1, blob_test)

    available_data_expand = expand_bitmap_white_area(available_data, 1)

    checkerboard_mask = numpy_checkerboard(image_shape)

    checkerboard_mask_inv = np.subtract(1, checkerboard_mask)

    available_data_v1 = np.multiply(checkerboard_mask, available_data)

    scrub_down_target = np.multiply(checkerboard_mask_inv, available_data_expand)

    available_data_point_set = np.where(available_data_v1==1)

    base_key_points = [available_data_point_set[i][:random_key_pixels_assigned] for i in range(2)]

    message_points = [available_data_point_set[i][random_key_pixels_assigned:] for i in range(2)]

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

    return img_step_2

def image_read_processing(image, base_key=None, string_shuffle_key=None,
                           custom_key_shuffle_key=None, random_key_pixels_assigned=None):
    key_max = 4294967295

    if base_key is None:
        base_key = 69298381

    if string_shuffle_key is None:
        string_shuffle_key = 58478232

    if custom_key_shuffle_key is None:
        custom_key_shuffle_key = 8274810

    if random_key_pixels_assigned is None:
        random_key_pixels_assigned = 400

    base_key %= key_max
    string_shuffle_key %= key_max
    custom_key_shuffle_key %= key_max

    image_shape = np.shape(image)

    image_shape = image_shape[:2]

    pool_test, scrub_test = full_image_blob_map(image, threshold=70, gap=30)

    blob_test = expand_bitmap_white_area(pool_test, 5)

    available_data = np.subtract(1, blob_test)

    checkerboard_mask = numpy_checkerboard(image_shape)

    available_data_v1 = np.multiply(checkerboard_mask, available_data)

    available_data_point_set = np.where(available_data_v1 == 1)

    base_key_points = [available_data_point_set[i][:random_key_pixels_assigned] for i in range(2)]

    message_points = [available_data_point_set[i][random_key_pixels_assigned:] for i in range(2)]

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