(declare-fun x () Real)
(declare-fun y () Real)
(assert 
(not (and (= (+ x y) 0.0) (= (- x y) 0.0)))
)
