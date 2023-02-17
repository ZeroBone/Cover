(declare-fun y () Real)
(declare-fun x () Real)
(assert (or (and (>= x 0.0) (<= x 10.0) (>= y 0.0) (<= y 10.0))
    (and (>= x 1.0) (<= x 11.0) (>= y 1.0) (<= y 11.0))
    (and (>= x 2.0) (<= x 12.0) (>= y 2.0) (<= y 12.0))))

