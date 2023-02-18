(declare-fun x () Real)
(declare-fun y () Real)
(declare-fun z () Real)
(assert 
(let ((a!1 (or (= (+ x y (* 2.0 z)) 0.0) (distinct (+ x y (* 2.0 z)) 0.0))))
  (and a!1 (= z 0.0)))
)
