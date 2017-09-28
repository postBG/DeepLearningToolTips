# Variables

A TensorFlow **variable** is the best way to represent shared, persistent state manipulated by your program.



`tf.Variable`은 op를 적용하므로써 값이 변화할 수 있는 tensor를 나타낸다. 또 `tf.Tensor`와는 다르게 `tf.Variable`은 특정한 `session.run`의 context에 속해있지 않다.



내부적으로 `tf.Variable`은 persistent tensor를 가지고 있다. 특정 op는 이 tensor의 값을 읽거나 고칠 수 있고, 이렇게 변화된 사항은 여러 `tf.Session`이 볼 수 있다.



## Creating a Variable

변수를 생성하는 가장 좋은 방법은 `tf.get_variable`함수를 사용하는 것이다. 이 함수는 name을 입력받는데, 이 이름은 여러 replica들이 이름을 통해 이 값에 접근하고, exporting과 checkpointing에 사용할 수 있도록 해준다. 

또 `tf.get_variable`을 이용하면 이전에 만들어진 이름이 같은 변수를 재사용해서 layer를 재사용하는 모델을 쉽게 정의할 수도 있다.



가장 단순하게 변수를 정의하기 위해서는 name과 shape를 넘겨주면 된다.

```python
my_variable = tf.get_variable("my_variable", [1, 2, 3])
```

이렇게 만들어진 변수는 default로 `tf.glorot_uniform_initializer`를 이용해 초기화된 tensor가 된다.



`dtype`을 선택적으로 명시할 수도 있다.

```python
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, 
  initializer=tf.zeros_initializer)
```



또한 특정값으로 tensor를 초기화할 수도 있다. 주의할 점은 이때는 tensor가 직접 initializer에 입력받은 tensor의 shape를 이용하기 때문에 shape를 입력해서는 안된다.

```python
other_variable = tf.get_variable("other_variable", dtype=tf.int32, 
  initializer=tf.constant([23, 42]))
```



### Variable collections

Tensorflow program의 disconnected된 부분에서 variable을 생성하고 싶은 경우가 있을 수 있기 때문에, 모든 변수에 접근할 수 있는 방법이 가끔 필요할 수도 있다. 이것을 위해 Tensorflow는 **collections**라는 것을 제공해주는데, **collections**는 tensor나 variable같은 object를 이름과 함께 저장해둔 리스트이다.



모든 `tf.Variable`은 기본적으로 아래 두가지 collection에 추가된다.

* `tf.GraphKeys.GLOBAL_VARIABLES`
  * variables that can be shared across multiple devices
* `tf.GraphKeys.TRAINABLE_VARIABLES`
  * variables for which TensorFlow will calculate gradients



만일 variable이 trainable하지 않게 만들고 싶다면, 아래와 같이 `tf.GraphKeys.LOCAL_VARIABLES`에 variable을 추가하면 된다.

```python
my_local = tf.get_variable("my_local", shape=(), 
collections=[tf.GraphKeys.LOCAL_VARIABLES])
```

혹은 아래와 같이 **trainable=False**를 줄 수도 있다.

```python
my_non_trainable = tf.get_variable("my_non_trainable", 
                                   shape=(), 
                                   trainable=False)
```



물론 `tf.add_to_collection`를 이용해 직접 만든 collection에 담을 수도 있다. collection은 명시적으로 생성할 필요 없이 string으로 된 이름을 이용하면 된다.

```pytho
tf.add_to_collection("my_collection_name", my_local)
```



조회는 아래와 같이 실행

```python
tf.get_collection("my_collection_name")
```



### Device placement

생략 [link](https://www.tensorflow.org/programmers_guide/variables#device_placement)



## Initializing variables

low-level Tensorflow API를 사용하는 경우에 variable을 사용하려면 직접 초기화해줘야 한다. 하지만 `tf.contrib.slim`나 `tf.estimator.Estimator`, `Keras` 같은 High-level API는 자동으로 변수를 초기화해준다.



물론 직접 손으로 하는 초기화가 귀찮기만한 과정은 아닌게, checkpoint로부터 pretrained된 모델을 로드한다면 비교적 비싼 비용의 연산은 초기화를 생략할 수도 있다.



사용하는 모든 variable들은 한번에 초기화하고 싶다면, training을 시작하기 전에 `tf.global_variables_initializer()`를 호출하면 된다. 이 함수는 `tf.GraphKeys.GLOBAL_VARIABLES` 콜렉션에 존재하는 모든 variable을 초기화하는 operation을 반환한다. 따라서 아래와 같이 초기화하면 된다.

```python
session.run(tf.global_variables_initializer())
# Now all variables are initialized.
```



주의해야할 점은 `tf.global_variables_initializer()`은 variable이 초기화되는 순서에 대한 규약은 없기 때문에

서로 의존관계가 있는 variable들의 경우, 초기화 중에 오류가 발생할 수 있다.

따라서 모든 variable이 초기화되지 않은 상태에서 variable의 vaule를 사용해야하는 상황에서는 `variable`대신 `variable.initialized_value()`를 사용하는 것이 좋다.

```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
```



상황에 따라 variable을 프로그래머가 직접 초기화하고 싶을 때도 있을 수 있는데, 이때는 variable의 initializer를 실행하면 된다.

```python
session.run(my_variable.initializer)
```



또한,아래와 같이 아직 초기화되지 않은 variable들을 모아볼 수도 있다.

```python
print(session.run(tf.report_uninitialized_variables()))
```



## Using variables

`tf.Variable`를 사용할 때에는 그냥 `tf.Tensor`를 다루듯이 하면 된다.

```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.
```



variable에 값을 대입하는 경우에는 `assign`, `assign_add`, 그리고 `tf.Variable`에 있는 것들을 사용하면 된다.

```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
tf.global_variables_initializer().run()
assignment.run()
```



variable들은 값이 바뀌기 때문에 한 시점에 어떤 버전의 값이 사용되었는지 아는 것이 유용할 수도 있다. variable에 어떤 operation을 한 이후에 값을 다시 읽고 싶을 때는 아래와 같이 `tf.Variable.read_value`를 사용하면 된다.

```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
  w = v.read_value()  # w is guaranteed to reflect v's value after the
                      # assign_add operation.
```



## Sharing Variable

Tensorflow는 variable을 공유하는 방법으로 다음 두가지를 제공한다:

1. `tf.Variable` object를 명시적으로 넘긴다.
   * 아주 명확하다는 장점이 있음
2. `tf.Variable` object를  `tf.variable_scope`를 이용해 암시적으로 wrapping한다.
   * 이 방식이 더 좋을 때가 많고, 대부분의 High-level API는 내부적으로 이러한 방식을 사용



**Variable scope**는 variable을 암묵적으로 생성하고 사용하는 함수를 호출하는 경우에 variable을 control할 수 있게 해준다. 또한 variable의 이름을 계층적이고 이해하기 쉽게 정의할 수 있게 해준다.



아래와 같이 conv layer를 정의하는 함수가 있다고 하자:

```python
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
```

위의 코드만 보면 variable에 "weights"와 같은 짧은 이름을 써서 명확해보인다. 하지만 실제 모델에서는 아래의 코드와 같이 이런 layer를 여러 개 사용하는 경우가 많아 저 함수를 여러번 호출하게 되는데 이때 문제가 발생한다.

```python
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 1, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
```

왜냐하면 아래의 코드가 같은 이름을 가진 새로운 variable을 생성해야할 지, 아니면 기존의 variable을 재사용해야할지 모호하기 때문이다. 이를 해결하기 위해 아래와 같이 `tf.variable_scope`를 사용할 수 있다.

만약에 의도가 새로운 variable을 생성하여 새 layer를 추가하는 것이었다면 아래와 같이 하면 된다.

```python
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 1, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```



반면에 이미 만들어진 variable을 재사용하는 것이 목적이라면 아래와 같이 `reuse=True`를 추가로 넘겨주면 된다.

```python
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)
```

혹은 위의 코드에서 `"model"`이라는 문자열을 반복하는 것이 싫다면 `scope.reuse_variables()`를 사용하거나 `tf.variable_scope`에 scope를 명시적으로 넘겨서 사용할 수도 있다.

```python
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)
```

or

```python
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)
```

