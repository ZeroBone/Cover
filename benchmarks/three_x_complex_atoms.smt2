(declare-fun x () Real)
(declare-fun y () Real)
(assert 
(let ((a!1 (or (= (- x (* 2.0 y)) 2.0) (distinct (- x (* 2.0 y)) 2.0)))
      (a!2 (not (and (= (+ x y) 0.0) (= (- x y) 0.0)))))
  (and a!1 a!2))
)
