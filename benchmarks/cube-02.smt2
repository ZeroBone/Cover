(declare-fun y () Real)
(declare-fun x () Real)
(assert (or (and (>= x 0.0) (<= x 10.0) (>= y 0.0) (<= y 10.0))
    (and (>= x 1.0) (<= x 11.0) (>= y 1.0) (<= y 11.0))))

