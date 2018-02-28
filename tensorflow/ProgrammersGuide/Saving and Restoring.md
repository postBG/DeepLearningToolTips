# Saving and Restoring

## Saving and restoring variables

`tf.train.Saver`가 saving과 restoring에 관한 메소드들을 제공한다.  `tf.train.Saver`의 생성자는 graph의 variable들이나 혹은 특정 variable들에게 `save`와  `restore` ops를 넣는다. `Saver` 객체는 이러한 ops를 실행하거나 어느 디렉토리에 checkpoint file들을 쓰고 읽을지 등을 조작할 수 있는 메소드를 제공한다.



Tensorflow는 **checkpoint file**에 binary 형태로 variable들을 저장하는데, 이는 간단히 말해 variable이름과 값을 mapping 해둔 파일이다.



### Saving Variables

`Saver` 객체를 생성하면 생성한 graph상의 모든 variable들을 다루는 객체가 생성되고, 여기에 `tf.train.Saver.save`함수를 이용해 값들을 저장한다.

```python
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
```



### Restoring Variables

`Saver`는 이름과는 달리 **checkpoint file**로부터 값을 읽어들여 restoring하는 역할까지 담당한다. 값을 복원하는 경우 graph의 variable들의 값들을 초기화하지 않아도 된다.

```python
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

