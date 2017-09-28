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

