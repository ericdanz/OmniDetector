import tensorflow as tf
import inception.ops_relu as ops
import inception.scopes as scopes
import numpy as np

SEED=123

def huber_loss(loss_tensor,delta=1):
    abs_loss_tensor = tf.abs(loss_tensor)
    less_than_tensor = 0.5 * tf.pow(abs_loss_tensor,2)
    greater_than_tensor = delta*(abs_loss_tensor-0.5*delta)
    return_tensor = tf.to_float(tf.less_equal(abs_loss_tensor,delta)) * less_than_tensor
    return_tensor += tf.to_float(tf.greater(abs_loss_tensor,delta)) * greater_than_tensor
    return return_tensor

def l2_scale(feature_maps,name):
    #this will l2 norm the feature maps individually, then figure a scale factor
    #assumes a 4D feature map, bxhxwxc
    with tf.name_scope(name) as scope:
        input_shape = feature_maps.get_shape()
        gamma_vals = np.ones((1,1,1,input_shape[3].value)) * 20

        gamma = tf.Variable(gamma_vals,name="gamma",dtype=tf.float32,collections=[tf.GraphKeys.WEIGHTS,tf.GraphKeys.VARIABLES])

        normed_feature_map = tf.nn.l2_normalize(feature_maps,dim=3) + 1e-8
        # scaled_normed_feature_map = tf.div(feature_maps,normed_feature_map)
        scaled_normed_feature_map = tf.mul(gamma,tf.div(feature_maps,normed_feature_map))

    return scaled_normed_feature_map


def bbox_recognition(input_feature_map,level_down_feature_map,name,padding="SAME",num_default_boxes=6,ksize=3,embedding_length=128,classes=80):
    #assuming 4D input feature_map
    input_shape = input_feature_map.get_shape()
    num_feature_maps = input_shape[3].value
    num_lower_feature_maps = level_down_feature_map.get_shape()[3].value
    #four localization parameters for the default boxes
    with tf.name_scope(name+'-lower3') as scope:
        bbox_lower_recognition_weights = tf.Variable(
            tf.truncated_normal([3, 3, num_lower_feature_maps, 128],
                                stddev=0.1,
                                seed=SEED),name="weights",collections=[tf.GraphKeys.WEIGHTS,tf.GraphKeys.VARIABLES,"recognition_vars"])
        bbox_lower_recognition_conv = tf.nn.conv2d(level_down_feature_map,
                            bbox_lower_recognition_weights,
                            strides=[1, 2, 2, 1],
                            padding=padding)
        bbox_lower_recognition_biases = tf.Variable(tf.zeros([128]),name="biases",collections=[tf.GraphKeys.BIASES,tf.GraphKeys.VARIABLES,"recognition_vars"])
        lower_output = tf.nn.elu(tf.nn.bias_add(bbox_lower_recognition_conv,bbox_lower_recognition_biases))

    with tf.name_scope(name) as scope:
        bbox_recognition_weights = tf.Variable(
            tf.truncated_normal([ksize, ksize, num_feature_maps+128, num_default_boxes*embedding_length],
                                stddev=0.1,
                                seed=SEED),name="weights",collections=[tf.GraphKeys.WEIGHTS,tf.GraphKeys.VARIABLES,"recognition_vars"])
        input_with_lower = tf.concat(3,[lower_output,input_feature_map])
        bbox_recognition_conv = tf.nn.conv2d(input_with_lower,
                            bbox_recognition_weights,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        bbox_recognition_biases = tf.Variable(tf.zeros([num_default_boxes*embedding_length]),name="biases",collections=[tf.GraphKeys.BIASES,tf.GraphKeys.VARIABLES,"recognition_vars"])
        embedding_output = tf.nn.bias_add(bbox_recognition_conv,bbox_recognition_biases)

    with tf.name_scope(name+'-class') as scope:
        bbox_recognition_weights = tf.Variable(
            tf.truncated_normal([ksize, ksize, num_default_boxes*embedding_length,num_default_boxes*classes],
                                stddev=0.1,
                                seed=SEED),name="weights",collections=[tf.GraphKeys.WEIGHTS,tf.GraphKeys.VARIABLES,"recognition_class_vars"])

        bbox_recognition_conv = tf.nn.conv2d(tf.nn.elu(embedding_output),
                            bbox_recognition_weights,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        bbox_recognition_biases = tf.Variable(tf.zeros([num_default_boxes*classes]),name="biases",collections=[tf.GraphKeys.BIASES,tf.GraphKeys.VARIABLES,"recognition_class_vars"])
        output = tf.nn.bias_add(bbox_recognition_conv,bbox_recognition_biases)

    return output,embedding_output


def inception_ssd(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 embedding_size=128,
                 is_training=True,
                 restore_logits=True,
                 batch_norm_params=None,
                 scope=''):

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.op_scope([inputs], scope, 'inception_v3'):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='VALID'):
        # 299 x 299 x 3
        end_points['conv0'] = ops.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0',batch_norm_params=batch_norm_params)
        tf.add_to_collection('debug',end_points['conv0'])

        # 149 x 149 x 32
        end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1',batch_norm_params=batch_norm_params)
        # 147 x 147 x 32
        end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3],
                                         padding='SAME', scope='conv2',batch_norm_params=batch_norm_params)
        # 147 x 147 x 64
        end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                           stride=2, scope='pool1')
        # 73 x 73 x 64
        end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1],
                                         scope='conv3',batch_norm_params=batch_norm_params)
        # 73 x 73 x 80.
        end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3],
                                         scope='conv4',batch_norm_params=batch_norm_params)
        # 71 x 71 x 192.
        end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                           stride=2, scope='pool2')
        # 35 x 35 x 192.
        net = end_points['pool2']
      # Inception blocks
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='SAME'):
        # mixed: 35 x 35 x 256.
        with tf.variable_scope('mixed_35x35x256a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1],batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1],batch_norm_params=batch_norm_params)
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5],batch_norm_params=batch_norm_params)
            tf.add_to_collection('debug',branch5x5)
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1],batch_norm_params=batch_norm_params)
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],batch_norm_params=batch_norm_params)
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 32, [1, 1],batch_norm_params=batch_norm_params)
          net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x256a'] = net
        # mixed_1: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1],batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1],batch_norm_params=batch_norm_params)
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5],batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1],batch_norm_params=batch_norm_params)
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],batch_norm_params=batch_norm_params)
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1],batch_norm_params=batch_norm_params)
          net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288a'] = net
        # mixed_2: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1],batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1],batch_norm_params=batch_norm_params)
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5],batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1],batch_norm_params=batch_norm_params)
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],batch_norm_params=batch_norm_params)
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1],batch_norm_params=batch_norm_params)
          net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288b'] = net
        # mixed_3: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID',batch_norm_params=batch_norm_params)
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1],batch_norm_params=batch_norm_params)
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],batch_norm_params=batch_norm_params)
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                      stride=2, padding='VALID',batch_norm_params=batch_norm_params)
          net = tf.concat(3, [branch3x3, branch3x3dbl])
          end_points['mixed_17x17x480a'] = net
        with tf.variable_scope('new_top8x8'): #5x5 if 200
            with tf.variable_scope('conv9'):
                conv9_1 = ops.conv2d(net, 64, [1, 1],collection=None,batch_norm_params=batch_norm_params)
                conv9_2 = ops.conv2d(conv9_1, 96, [3, 3], stride=2, padding='SAME',collection=None,batch_norm_params=batch_norm_params)
        with tf.variable_scope('new_top4x4'): #2x2 if 200
            with tf.variable_scope('conv10'):
                conv10_1 = ops.conv2d(conv9_2, 64, [1, 1],collection=None,batch_norm_params=batch_norm_params)
                conv10_2 = ops.conv2d(conv10_1, 96, [3, 3], stride=2, padding='SAME',collection=None,batch_norm_params=batch_norm_params)

        with tf.variable_scope('new_top1'):
            conv10_2_shape = conv10_2.get_shape()
            pool_final = tf.nn.avg_pool(conv10_2,ksize=[1, conv10_2_shape[1].value,
                    conv10_2_shape[2].value, 1], strides=[1, conv10_2_shape[1].value, conv10_2_shape[2].value,1],
                    padding="VALID",name="pool_final") #could change to make a smaller pool output



        conv9_2_bbox_classification,conv9_2_bbox_embedding = bbox_recognition(conv9_2,end_points['mixed_17x17x480a'],"conv9_2_bbox_embedding",num_default_boxes=3,embedding_length=embedding_size,classes=num_classes)

        conv10_2_bbox_classification,conv10_2_bbox_embedding = bbox_recognition(conv10_2,conv9_2,"conv10_2_bbox_embedding",num_default_boxes=3,embedding_length=embedding_size,classes=num_classes)
        tf.add_to_collection('debug',conv10_2_bbox_embedding)

        pool_final_bbox_classification,pool_final_bbox_embedding = bbox_recognition(pool_final,conv10_2,"pool_final_bbox_embedding",padding="VALID",ksize=1,num_default_boxes=3,embedding_length=embedding_size,classes=num_classes)
        tf.add_to_collection('debug',pool_final_bbox_embedding)


        #put them all together to return their values
        #reshape them into lists defaultboxes*h*w long

        original_feature_shape = conv9_2_bbox_classification.get_shape()

        reshaped_conv9_2_bbox_classification = tf.reshape(conv9_2_bbox_classification,[original_feature_shape[0].value,-1,num_classes])
        reshaped_conv9_2_bbox_embedding = tf.reshape(conv9_2_bbox_embedding,[original_feature_shape[0].value,-1,embedding_size])

        reshaped_conv10_2_bbox_classification = tf.reshape(conv10_2_bbox_classification,[original_feature_shape[0].value,-1,num_classes])
        reshaped_conv10_2_bbox_embedding = tf.reshape(conv10_2_bbox_embedding,[original_feature_shape[0].value,-1,embedding_size])

        reshaped_pool_final_bbox_classification = tf.reshape(pool_final_bbox_classification,[original_feature_shape[0].value,-1,num_classes])
        reshaped_pool_final_bbox_embedding = tf.reshape(pool_final_bbox_embedding,[original_feature_shape[0].value,-1,embedding_size])

        #then concatenate them
        classification_output = tf.concat(1,[reshaped_conv9_2_bbox_classification,reshaped_conv10_2_bbox_classification,
                                reshaped_pool_final_bbox_classification])

        embedding_output = tf.concat(1,[reshaped_conv9_2_bbox_embedding,reshaped_conv10_2_bbox_embedding,
                                reshaped_pool_final_bbox_embedding])

        feature_map_size_list = [[conv9_2_bbox_embedding.get_shape()[1].value,conv9_2_bbox_embedding.get_shape()[2].value],
                                [conv10_2_bbox_embedding.get_shape()[1].value,conv10_2_bbox_embedding.get_shape()[2].value],
                                [pool_final_bbox_embedding.get_shape()[1].value,pool_final_bbox_embedding.get_shape()[2].value]]


        return classification_output,embedding_output,feature_map_size_list


def calculate_detection_loss(confidence_output,localization_output,confidence_labels,localization_labels,sparse_labels):
    #expects everything to have 3D shape
    #batch, box_list, class or localization

    N = tf.reduce_sum(tf.to_float(tf.not_equal(confidence_labels,0))) + 1
    # N = tf.to_float(tf.less(N,1)) * 10

    #make all tensors 2d
    confidence_output_shape = confidence_output.get_shape()

    flattened_confidence_output = tf.reshape(confidence_output,[-1,confidence_output_shape[2].value])
    flattened_confidence_labels = tf.reshape(confidence_labels,[-1])

    flattened_localization_output = tf.reshape(localization_output,[-1,4])
    flattened_localization_labels = tf.reshape(localization_labels,[-1,4])

    negative_indices = tf.to_float(tf.equal(flattened_confidence_labels,0))

    softmaxed_flattened_confidence_output = tf.nn.softmax(tf.identity(flattened_confidence_output))
    negative_output_values = softmaxed_flattened_confidence_output[:,0]

    inverted_values = (tf.ones_like(negative_output_values)-negative_output_values)
    masked_negative_output_values = tf.reshape(inverted_values * negative_indices,[-1])

    values,indices = tf.nn.top_k(masked_negative_output_values,tf.to_int32(N*3)) #hard negative mining, limited to 3x positive labels
    sparse_labels_update = tf.reshape(flattened_confidence_labels,[-1]) #take the positive labels
    sparse_labels_update2 = tf.add(sparse_labels_update,tf.to_float(tf.equal(sparse_labels_update,0)) * -1) #set all 0's (negatives) to -1

    set_op = sparse_labels.assign(sparse_labels_update2)
    sparse_labels_with_hard_negatives = tf.scatter_update(set_op,indices,tf.zeros_like(values)) #make hard negatives 0 again

    #localization
    combined_localization_labels = tf.identity(flattened_localization_output) * tf.to_float(tf.equal(flattened_localization_labels,0)) + flattened_localization_labels

    #loss
    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(flattened_confidence_output, tf.to_int32(sparse_labels_with_hard_negatives), name="xentropy")
    # class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(flattened_confidence_output, tf.to_int32(tf.ones_like(sparse_labels)*-1), name="xentropy")
    class_loss_zeroed = tf.mul(class_loss,tf.to_float(tf.not_equal(class_loss,0))) #multiply by a number that isn't itself

    localization_loss = huber_loss(flattened_localization_output-combined_localization_labels)

    loss = tf.div(tf.reduce_sum(class_loss_zeroed) + tf.reduce_sum(localization_loss),(N*4+1))
    reg_weights = 1e-4*tf.pack([tf.nn.l2_loss(i) for i in tf.get_collection(tf.GraphKeys.WEIGHTS)])
    reg_loss = tf.reduce_sum(reg_weights)

    loss += reg_loss
    # loss += verification_loss #wait until the face parts converge
    # loss = tf.div( tf.reduce_sum(localization_loss),(N*4+1))


    return loss,sparse_labels


def calculate_verification_loss(verification_output,verification_labels,sparse_labels):

    flattened_verification_output = tf.reshape(verification_output,[-1,verification_output.get_shape()[2].value]) #or whatever the dimension winds up being
    flattened_verification_output = tf.nn.l2_normalize(flattened_verification_output,1)
    flattened_verification_labels = tf.reshape(verification_labels,[-1,verification_labels.get_shape()[2].value])
    flattened_verification_labels += tf.reshape(tf.to_float(tf.equal(tf.reduce_sum(flattened_verification_labels,1),0)),[-1,1])*flattened_verification_output
    verification_loss = tf.reduce_sum(tf.nn.l2_loss(verification_labels-verification_output))
    return verification_loss
