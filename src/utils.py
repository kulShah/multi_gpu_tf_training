import tensorflow as tf

def average_gradients(tower_grads):
  
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # note that each grad_and_vars looks like :
    # ( (grad0_gpu_0), ... , (grad_n, gpu_n) )
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower
      expanded_g = tf.expand_dims(g, 0)

      # append on a 'tower' dimension which we will average over below
      grads.append(expanded_g)

    # average over the 'tower' dimension
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # the variables are redundant because they are shared across towers
    # so, just return the first tower's pointer to the Variable

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)

  return average_grads


# by default, all variables will be placed on '/gpu:0'
# we need a custom device function to assign all the variables to '/cpu:0'
# note: if GPUs are peered, '/gpu:0' can be a faster option

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
  def _assign(op):
    node_def = op if isinstance(op, tf.NodeDef) else op.node_def
    if node_def.op in PS_OPS:
      return "/" + ps_device
    else:
      return device

  return _assign

  