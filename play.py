import tensorflow as tf

state = tf.Variable(0, name="counter")
added = tf.placeholder(tf.int32)
new_value = tf.add(state, added)
update = tf.assign(state, new_value)
init_op = tf.initialize_all_variables()

feed = {
    added: 3,
}
with tf.Session() as sess:
        sess.run(init_op, feed)
        print(sess.run(state, feed))
        for _ in range(3):
                sess.run(update, feed)
                print(sess.run(state, feed))
