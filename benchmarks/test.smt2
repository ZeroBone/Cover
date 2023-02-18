(declare-fun x () Real)
(declare-fun y () Real)
(assert 
(let ((a!1 (or (and (<= x (- 2.0)) (> y 1.0) (<= (- x y) 1.0))
               (and (= x (- 2.0)) (>= y 1.0) (<= (- x y) 1.0))
               (and (<= y 1.0) (>= x 2.0) (>= (- x y) 1.0))
               (and (>= y 1.0) (> x 2.0)))))
  (not a!1))
)
