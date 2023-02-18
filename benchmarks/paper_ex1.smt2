(set-info :comment "(and (>= x 0) (>= (+ (* 3 y) (* 9 z)) 0) (= (+ y (* 3 z)) 7))")
(set-info :comment "decomposable for {x},{y,z} but not for {{x,y,z}}")
(declare-fun x () Real)
(declare-fun y () Real)
(declare-fun z () Real)
(assert
(and (>= x 0) (>= (+ (* 3 y) (* 9 z)) 0) (= (+ y (* 3 z)) 7))
)

