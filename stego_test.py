import stego
import cv2
import numpy as np
import sys

def gen_old_diamond(max_size):
    pattern_top = [[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1]]
    pattern_bottom = [[0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]]
    pattern_top = np.array(pattern_top)
    pattern_bottom = np.array(pattern_bottom)
    rows_to_add = (max_size - 13) // 5
    pattern_middle = np.tile([1, 1, 1, 1, 1], (rows_to_add, 1))
    if rows_to_add <= 0:
        pattern_template = np.concatenate((pattern_top, pattern_bottom), axis=0)
    else:
        pattern_template = np.concatenate((pattern_top, pattern_middle, pattern_bottom), axis=0)
    return pattern_template

def generate_pattern(shape, max_size):
    pattern_top = [[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1]]
    pattern_bottom = [[0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]]
    pattern_top = np.array(pattern_top)
    pattern_bottom = np.array(pattern_bottom)
    rows_to_add = (max_size - 13) // 5

    t_grid = np.zeros((10*(5+rows_to_add),50))

    pattern_middle = np.tile([1, 1, 1, 1, 1], (rows_to_add, 1))

    if rows_to_add <= 0:
        pattern_template = np.concatenate((pattern_top, pattern_bottom), axis=0)
    else:
        pattern_template = np.concatenate((pattern_top, pattern_middle, pattern_bottom), axis=0)

    half_increments = [rows_to_add+3, 3]
    full_shape = list(np.shape(pattern_template))
    full_increments = np.copy(full_shape)
    full_increments[0] += rows_to_add+1
    full_increments[1] += 1

    grid = np.zeros(shape)

    for y in range(1,5):
        for x in range(1,5):
            c_1 = y*full_increments[0]
            r_1 = x*full_increments[1]
            c_2 = c_1+half_increments[0]
            r_2 = r_1+half_increments[1]
            test_select = grid[c_1:c_1+full_shape[0]+1,r_1:r_1+full_shape[1]+1]
            t_grid[c_1:c_1+full_shape[0],r_1:r_1+full_shape[1]] = np.maximum(t_grid[c_1:c_1+full_shape[0],r_1:r_1+full_shape[1]], pattern_template)
            t_grid[c_2:c_2+full_shape[0],r_2:r_2+full_shape[1]] = np.maximum(t_grid[c_2:c_2+full_shape[0],r_2:r_2+full_shape[1]], pattern_template)
    
    start_col_position = 2*full_increments[0]
    start_row_position = 2*full_increments[1]
    
    grid_template = np.copy(t_grid[start_col_position:start_col_position+6+2*rows_to_add,start_row_position:start_row_position+6])
    template_shape = list(np.shape(grid_template))

    shape_multiples = list(np.shape(grid_template))

    temp_shape_adjust = list(shape) #multiple of
    temp_shape_adjust[0] += 3*shape_multiples[0] - (temp_shape_adjust[0]%shape_multiples[0])
    temp_shape_adjust[1] += 3*shape_multiples[1] - (temp_shape_adjust[1]%shape_multiples[1])

    grid = np.zeros(tuple(temp_shape_adjust))

    for col in range(temp_shape_adjust[0] // shape_multiples[0]):
        for row in range(temp_shape_adjust[1] // shape_multiples[1]):
            col_pos = col*shape_multiples[0]
            row_pos = row*shape_multiples[1]
            grid[col_pos:col_pos+shape_multiples[0],row_pos:row_pos+shape_multiples[1]] = np.maximum(grid[col_pos:col_pos+shape_multiples[0],row_pos:row_pos+shape_multiples[1]], grid_template)
    

    return grid[0:shape[0],0:shape[1]]

def compare_blobs():
    #img = cv2.imread("C:/Users/subje/Downloads/dhop.jpg")
    img = cv2.imread("C:/Users/subje/Downloads/JPow.jpg")
    lsb_layer = stego.isolate_bit_image(img, 7)
    image_size_assigned = stego.image_size_assignment(lsb_layer)

    threshold = 70
    gap = 5

    print("image_size_assigned")
    print(image_size_assigned)

    pool_test_2 = np.zeros_like(lsb_layer)
    scrub_test_2 = np.copy(pool_test_2)
    pool_test_2[np.where(image_size_assigned >= threshold)] = 1
    scrub_test_2[np.where(image_size_assigned >= threshold-gap)] = 1
    scrub_test_2 = scrub_test_2*(1-pool_test_2)

    pool_test, scrub_test = stego.full_image_blob_map(img, threshold=threshold, gap=gap)

    #pool_diff = np.where((pool_test != pool_test_2))
    pool_diff_map = pool_test*(1-pool_test_2)
    #pool_diff_map[pool_diff] = 1

    #scrub_diff = np.where((scrub_test != scrub_test_2))
    scrub_diff_map = scrub_test*(1-scrub_test_2)
    #scrub_diff_map[scrub_diff] = 1

    cv2.imshow("scrub_difference", scrub_diff_map.astype('float'))
    cv2.imshow("pool_1", pool_test.astype('float'))
    cv2.imshow("pool_2", pool_test_2.astype('float'))
    cv2.imshow("LSB", stego.convert_255(stego.isolate_bit_image(img, 7, return_rgb_image=True)))
    cv2.waitKey(0)

def main_2():
    img = cv2.imread("C:/Users/subje/Downloads/dhop.jpg")
    #img = cv2.imread("C:/Users/subje/Downloads/JPow.jpg")
    alphabet = np.array([char for char in "abcdefghijklmnopqrstuvwxyz0123456789"])
    message_length = 20000
    letter_select = np.random.randint(len(alphabet), size=message_length)
    message = ""
    for char in alphabet[letter_select]:
        message += char

    img_written = image_write(img, message)
    

    message_recover = image_read(img_written)

    if (message == message_recover):
        print("message recovered exactly")

def main_3():
    img = cv2.imread("C:/Users/subje/Downloads/dhop.jpg")
    #img = cv2.imread("C:/Users/subje/Downloads/JPow.jpg")
    alphabet = np.array([char for char in "abcdefghijklmnopqrstuvwxyz0123456789"])
    message_length = 20000
    letter_select = np.random.randint(len(alphabet), size=message_length)
    message = ""
    for char in alphabet[letter_select]:
        message += char
    visual_get_0 = pool_mask_visual(stego.isolate_bit_image(img, 7))
    cv2.imwrite("lsb_original.png", stego.convert_255(stego.isolate_bit_image(img, 7, return_rgb_image=True)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite("mask_generated_original.png", visual_get_0, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    blob_size = 7
    threshold = 15

    img_written = stego.image_write_new(img, message, cover_flag=True, blob_expand_size=blob_size, threshold=threshold)
    
    visual_get = pool_mask_visual(stego.isolate_bit_image(img_written, 7))
    
    cv2.imwrite("lsb_written.png", stego.convert_255(stego.isolate_bit_image(img_written, 7, return_rgb_image=True)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite("mask_generated_written.png", visual_get, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    message_recover = stego.image_read_new(img_written, blob_expand_size=blob_size, threshold=threshold)

    if (message == message_recover):
        print("message recovered exactly")

def pool_mask_visual(array_in, is_size_assignment=None):
    if is_size_assignment is None:
        is_size_assignment = False
    if is_size_assignment:
        float_mask = np.divide(np.sqrt(array_in).astype('float'), np.sqrt(np.max(array_in)))
    else:
        size_assignment = stego.image_size_assignment(array_in)
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

def main():
    img = cv2.imread("C:/Users/subje/Downloads/dhop.jpg")
    alphabet = np.array([char for char in "abcdefghijklmnopqrstuvwxyz0123456789"])
    message_length = 20000
    letter_select = np.random.randint(len(alphabet), size=message_length)
    message = ""
    for char in alphabet[letter_select]:
        message += char



    img_make = stego.image_write_processing_v2(img, message)

    message_recover = stego.image_read_processing_v2(img_make)

    if message == message_recover:
        print("Message recovered exactly")

def image_write(img_in, message, shuffle_key=None, threshold=None, cover_flag=None):

    img = np.copy(img_in)

    if threshold is None:
        threshold = 40
    if shuffle_key is None:
        shuffle_key = 17876418
    if cover_flag is None:
        cover_flag = False

    shuffle_key %= (2**32)-3
    recovery_key = np.sin(shuffle_key).astype('float64')
    recovery_key = recovery_key - np.floor(recovery_key)
    recovery_key = np.floor((10**9)*recovery_key)
    recovery_key = (recovery_key % ((2**32)-3)).astype('uint32')
    print("recovery_key ->",recovery_key)

    lsb_layer = stego.isolate_bit_image(img, 7)
    image_size_assigned = stego.image_size_assignment(lsb_layer)

    pool_test = np.zeros_like(lsb_layer)
    pool_test[np.where(image_size_assigned >= threshold)] = 1

    print("pool_hash_write ->",hash(pool_test.tobytes()))

    blob_test = stego.expand_bitmap_white_area(pool_test, 3)
    blob_test_2 = stego.expand_bitmap_white_area(blob_test, 2)

    scrub_area = blob_test_2*(1-blob_test)
    new_image = stego.sanitize_image(img, scrub_area)

    scrub_area_2 = np.zeros_like(lsb_layer)
    scrub_area_2[np.where(image_size_assigned >= threshold-5)] = 1
    scrub_area_2 = scrub_area_2*(1-pool_test)

    new_image = stego.sanitize_image(new_image, scrub_area_2)
    writable_pattern = generate_pattern(np.shape(lsb_layer), threshold-5)
    target_set = (1-blob_test_2)*writable_pattern

    print("total bits before pattern correction ->",np.sum((1-blob_test_2)))
    print("total bits after  pattern correction ->",np.sum(target_set))

    target_set = np.where(target_set == 1)
    target_set = np.array(list(target_set))

    for i in range(2):
        target_set[i] = stego.shuffle_seed(target_set[i], shuffle_key)

    scrub_set = (1-blob_test_2)*(1-writable_pattern)
    bit_make = stego.to_bits(message)
    bit_make = stego.string_shuffle_seed(bit_make, recovery_key)
    new_image = stego.bits_to_image(new_image, bit_make, target_set)

    #Both options below prevent pool corruption
    #However, the latter option makes it somewhat visible using the debuging methods shown here
    #The first option computes the pool size again. For large images, this time doubling will be noticable
    #Enabling cover flag makes the write much less detectable.

    if cover_flag: 
        size_get_step_3 = stego.image_size_assignment(stego.isolate_bit_image(new_image, 7))
        final_scrub_set = np.zeros_like(scrub_set)
        final_scrub_set[np.where(size_get_step_3 >= threshold)] = 1
        final_scrub_set = final_scrub_set*(scrub_set)
        new_image = stego.sanitize_image(new_image, final_scrub_set)
    else:
        new_image = stego.sanitize_image(new_image, scrub_set)
    return new_image

def image_read(img, shuffle_key=None, threshold=None):
    if threshold is None:
        threshold = 40
    if shuffle_key is None:
        shuffle_key = 17876418

    shuffle_key %= (2**32)-3
    recovery_key = np.sin(shuffle_key).astype('float64')
    recovery_key = recovery_key - np.floor(recovery_key)
    recovery_key = np.floor((10**9)*recovery_key)
    recovery_key = (recovery_key % ((2**32)-3)).astype('uint32')
    print("recovery_key ->",recovery_key)

    lsb_layer = stego.isolate_bit_image(img, 7)
    image_size_assigned = stego.image_size_assignment(lsb_layer)

    float_mask = np.divide(np.sqrt(image_size_assigned).astype('float'), np.sqrt(np.max(image_size_assigned)))
    cv2.imwrite("mask_generated_1.png", np.multiply(float_mask,255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

    print("image_size_assigned")
    print(image_size_assigned)

    pool_test = np.zeros_like(lsb_layer)
    pool_test[np.where(image_size_assigned >= threshold)] = 1

    print("pool_hash_read ->",hash(pool_test.tobytes()))

    blob_test = stego.expand_bitmap_white_area(pool_test, 3)
    blob_test_2 = stego.expand_bitmap_white_area(blob_test, 2)

    writable_pattern = generate_pattern(np.shape(lsb_layer), threshold-5)

    target_set = (1-blob_test_2)*writable_pattern

    print("total bits before pattern correction ->",np.sum((1-blob_test_2)))
    print("total bits after  pattern correction ->",np.sum(target_set))

    target_set = np.where(target_set == 1)
    target_set = np.array(list(target_set))
    for i in range(2):
        target_set[i] = stego.shuffle_seed(target_set[i], shuffle_key)

    bits_read = stego.bits_from_image(img, target_set)
    bits_read = stego.string_unshuffle_seed(bits_read, recovery_key)

    message_read = stego.from_bits(bits_read)

    return message_read

def pattern_test():
    tile_test = np.tile([1,2,3,4,5], (3, 1))
    print("tile_test")
    print(tile_test)
    
    
    diamond = gen_diamond(20)
    #np.set_printoptions(threshold=sys.maxsize)
    #np.set_printoptions(linewidth=np.inf)
    print("diamond")
    print(diamond)
    print("shape",np.shape(diamond))


    grid_get = generate_pattern((400, 400), 57)
    cv2.imwrite("pattern_made.png", np.multiply(grid_get, 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

def main_4():
    make_pattern = gen_exclusion_pattern((400, 400),81)
    cv2.imwrite("pattern_make.png", stego.convert_255(make_pattern), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    #diamond_make = gen_old_diamond(80)
    #diamond_make = gen_diamond(6)
    #tile_pattern = tile_diamond((400, 400), diamond_make)







#compare_blobs()
#main()
main_3()
#main_4()
#pattern_test()
