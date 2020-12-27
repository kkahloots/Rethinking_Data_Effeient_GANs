import numpy as np
import tensorflow as tf


def distort(images, num_anchors=10, perturb_sigma=5.0):
    # Similar results to elastic deformation (a bit complex transformation)
    # However, the transformation is much faster that elastic deformation and have a straightforward arguments
    # TODO: Need to adapt reflect padding and eliminate out-of-frame
    # images is 4D tensor [B,H,W,C]
    # num_anchors : the number of base position to make distortion, total anchors in a image = num_anchors**2
    # perturb_sigma : the displacement sigma of each anchor

    src_shp_list = images.get_shape().as_list()
    batch_size, src_height, src_width = tf.unstack(tf.shape(images))[:3]

    pad_size = tf.cast(tf.cast(tf.maximum(src_height, src_width), tf.float32) * (np.sqrt(2) - 1.0) / 2 + 0.5, tf.int32)
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    height, width = tf.unstack(tf.shape(images))[1:3]

    mapx_base = tf.matmul(tf.ones(shape=tf.stack([num_anchors, 1])),
                          tf.transpose(tf.expand_dims(tf.linspace(0., tf.cast(width, tf.float32), num_anchors), 1), [1, 0]))
    mapy_base = tf.matmul(tf.expand_dims(tf.linspace(0., tf.cast(height, tf.float32), num_anchors), 1),
                          tf.ones(shape=tf.stack([1, num_anchors])))

    mapx_base = tf.tile(mapx_base[None, ..., None], [batch_size, 1, 1, 1])  # [batch_size, N, N, 1]
    mapy_base = tf.tile(mapy_base[None, ..., None], [batch_size, 1, 1, 1])
    distortion_x = tf.random.normal((batch_size, num_anchors, num_anchors, 1), stddev=perturb_sigma)
    distortion_y = tf.random.normal((batch_size, num_anchors, num_anchors, 1), stddev=perturb_sigma)
    mapx = mapx_base + distortion_x
    mapy = mapy_base + distortion_y

    interp_mapx = tf.compat.v1.image.resize(mapx, size=(height, width), method=tf.image.ResizeMethod.BILINEAR,
                                         align_corners=True)
    interp_mapy = tf.compat.v1.image.resize(mapy, size=(height, width), method=tf.image.ResizeMethod.BILINEAR,
                                         align_corners=True)
    coord_maps = tf.concat([interp_mapx, interp_mapy], axis=-1)  # [batch_size, height, width, 2]

    warp_images = bilinear_sampling(images, coord_maps)

    warp_images = tf.slice(warp_images, [0, pad_size, pad_size, 0], [-1, src_height, src_width, -1])

    warp_images.set_shape(src_shp_list)

    return warp_images


def bilinear_sampling(photos, coords):
    """Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    Args:
        photos: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,
          width_t, 2]. height_t/width_t correspond to the dimensions of the output
          image (don't need to be the same as height_s/width_s). The two channels
          correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """

    # photos: [batch_size, height2, width2, C]
    # coords: [batch_size, height1, width1, C]
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, tf.float32)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = tf.shape(photos)
    coord_size = tf.shape(coords)

    out_size = tf.stack([coord_size[0],
                         coord_size[1],
                         coord_size[2],
                         inp_size[3],
                         ])

    coords_x = tf.cast(coords_x, tf.float32)
    coords_y = tf.cast(coords_y, tf.float32)

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(photos)[1] - 1, tf.float32)
    x_max = tf.cast(tf.shape(photos)[2] - 1, tf.float32)
    zero = tf.zeros([1], dtype=tf.float32)

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], tf.float32)
    dim1 = tf.cast(inp_size[2] * inp_size[1], tf.float32)
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), tf.float32) * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from photos
    photos_flat = tf.reshape(photos, tf.stack([-1, inp_size[3]]))
    photos_flat = tf.cast(photos_flat, tf.float32)

    im00 = tf.reshape(tf.gather(photos_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(photos_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(photos_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(photos_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    out_photos = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])

    return out_photos


#
# def distort(images, grid_width, grid_height, magnitude):
#     im_shape = tf.shape(images)
#     h, w = tf.unstack(im_shape)[1:3]
#
#     horizontal_tiles = grid_width
#     vertical_tiles = grid_height
#
#     width_of_square = int(floor(float(w) / float(horizontal_tiles)))
#     height_of_square = int(floor(float(h) / float(vertical_tiles)))
#
#     width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
#     height_of_last_square = h - (height_of_square * (vertical_tiles - 1))
#
#     dimensions = []
#
#     for vertical_tile in range(vertical_tiles):
#         for horizontal_tile in range(horizontal_tiles):
#             if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
#                 dimensions.append([horizontal_tile * width_of_square,
#                                    vertical_tile * height_of_square,
#                                    width_of_last_square + (horizontal_tile * width_of_square),
#                                    height_of_last_square + (height_of_square * vertical_tile)])
#             elif vertical_tile == (vertical_tiles - 1):
#                 dimensions.append([horizontal_tile * width_of_square,
#                                    vertical_tile * height_of_square,
#                                    width_of_square + (horizontal_tile * width_of_square),
#                                    height_of_last_square + (height_of_square * vertical_tile)])
#             elif horizontal_tile == (horizontal_tiles - 1):
#                 dimensions.append([horizontal_tile * width_of_square,
#                                    vertical_tile * height_of_square,
#                                    width_of_last_square + (horizontal_tile * width_of_square),
#                                    height_of_square + (height_of_square * vertical_tile)])
#             else:
#                 dimensions.append([horizontal_tile * width_of_square,
#                                    vertical_tile * height_of_square,
#                                    width_of_square + (horizontal_tile * width_of_square),
#                                    height_of_square + (height_of_square * vertical_tile)])
#
#     # For loop that generates polygons could be rewritten, but maybe harder to read?
#     # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]
#
#     # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
#     last_column = []
#     for i in range(vertical_tiles):
#         last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)
#
#     last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)
#
#     polygons = []
#     for x1, y1, x2, y2 in dimensions:
#         polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])
#
#     polygon_indices = []
#     for i in range((vertical_tiles * horizontal_tiles) - 1):
#         if i not in last_row and i not in last_column:
#             polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])
#
#     for a, b, c, d in polygon_indices:
#         dx = random.randint(-magnitude, magnitude)
#         dy = random.randint(-magnitude, magnitude)
#
#         x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
#         polygons[a] = [x1, y1,
#                        x2, y2,
#                        x3 + dx, y3 + dy,
#                        x4, y4]
#
#         x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
#         polygons[b] = [x1, y1,
#                        x2 + dx, y2 + dy,
#                        x3, y3,
#                        x4, y4]
#
#         x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
#         polygons[c] = [x1, y1,
#                        x2, y2,
#                        x3, y3,
#                        x4 + dx, y4 + dy]
#
#         x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
#         polygons[d] = [x1 + dx, y1 + dy,
#                        x2, y2,
#                        x3, y3,
#                        x4, y4]
#
#     generated_mesh = []
#     for i in range(len(dimensions)):
#         generated_mesh.append([dimensions[i], polygons[i]])
#
#     for box, quad in generated_mesh:
#         w = box[2] - box[0]
#         h = box[3] - box[1]
#
#         # quadrilateral warp.  data specifies the four corners
#         # given as NW, SW, SE, and NE.
#         nw = quad[0:2]
#         sw = quad[2:4]
#         se = quad[4:6]
#         ne = quad[6:8]
#
#         if hasattr(w, 'numpy'):
#             w = w.numpy()
#         if hasattr(h, 'numpy'):
#             h = h.numpy()
#
#         if hasattr(nw, 'numpy'):
#             nw = nw.numpy()
#         if hasattr(nw[0], 'numpy'):
#             nw[0] = nw[0].numpy()
#         if hasattr(nw[1], 'numpy'):
#             nw[1] = nw[1].numpy()
#
#         if hasattr(sw, 'numpy'):
#             sw = sw.numpy()
#         if hasattr(sw[0], 'numpy'):
#             sw[0] = sw[0].numpy()
#         if hasattr(sw[1], 'numpy'):
#             sw[1] = sw[1].numpy()
#
#         if hasattr(se, 'numpy'):
#             se = se.numpy()
#         if hasattr(se[0], 'numpy'):
#             se[0] = se[0].numpy()
#         if hasattr(se[1], 'numpy'):
#             se[1] = se[1].numpy()
#
#         if hasattr(ne, 'numpy'):
#             ne = ne.numpy()
#         if hasattr(ne[0], 'numpy'):
#             ne[0] = ne[0].numpy()
#         if hasattr(ne[1], 'numpy'):
#             ne[1] = ne[1].numpy()
#
#
#         x0, y0 = nw
#         As = 1.0 / w
#         At = 1.0 / h
#
#         data = (
#             x0,
#             (ne[0] - x0) * As,
#             (sw[0] - x0) * At,
#             (se[0] - sw[0] - ne[0] + x0) * As * At,
#             y0,
#             (ne[1] - y0) * As,
#             (sw[1] - y0) * At,
#             (se[1] - sw[1] - ne[1] + y0) * As * At,
#         )
#
#         images =  tfa.image.transform(images, data)
#     return images
#     #return tfa.image.transform(images[0], polygons)
#     #return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)
