(declare-fun x () Real)
(declare-fun y () Real)
(declare-fun z () Real)
(assert 
(let ((a!1 (or (= (+ (+ x y) (* 2.0 z)) 0.0)
               (distinct (+ (+ x y) (* 2.0 z)) 0.0)))
      (a!2 (and (= (+ (+ x y) (* 2.0 z)) 0.0) (= (+ x y) 0.0) (= (- x z) 0.0))))
  (and a!1
       (or (= (+ x y) 0.0) (distinct (+ x y) 0.0))
       (or (= (- x z) 0.0) (distinct (- x z) 0.0))
       (not a!2)))
)
