(declare-fun x () Real)
(declare-fun y () Real)
(declare-fun z () Real)
(assert 
(let ((a!1 (and (= (- x y) 0.0) (= (+ (- x y) z) 0.0))))
  (or a!1 (= z 0.0)))
)
