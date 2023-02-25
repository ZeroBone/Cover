(declare-fun y () Real)
(declare-fun x () Real)
(assert (or (and (>= x 0.0) (<= x 5.0) (>= y 0.0) (<= y 5.0))
    (and (>= x 1.0) (<= x 6.0) (>= y 1.0) (<= y 6.0))))
