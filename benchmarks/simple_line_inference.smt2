(declare-fun x () Real)
(declare-fun y () Real)
(assert 
(and (= y 10.0) (<= (+ x y) 15.0))
)
