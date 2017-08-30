# Graphs and Sessions

Tensorflow = data flow graph를 그리고 그것을 실행하기 위한 session을 생성해 실행

## Dataflow graph

Dataflow graph의 장점

*  **Parallelism.** 
  operation들간의 의존관계를 명시적으로 그래프로 나타냄으로써, 병렬로 실행가능한 부분이 어디인지 쉽게 판단할 수 있게 해준다.
* **Distributed execution.**
  위와 같은 이유로 데이터의 흐름을 그래프로 표현하므로써, 여러 디바이스(CPU, GPU, and TPU)에 적절하게 분리시킬 수 있다.
* **Compilation.**
  그래프를 분석하므로써 더 빠른 코드를 generate할 수 있다.
* **Portability.**
  dataflow graph represenation은 언어독립적이므로 서로 다른 언어로 작성할 수도 있고, 저장과 복구도 용이해진다.