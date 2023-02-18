(declare-fun x () Real)
(declare-fun y_1 () Real)
(declare-fun y_2 () Real)
(assert 
(and (< (- (+ y_1 y_2) x) 0.0)
     (> (- (+ y_1 y_2) x) -1.0)
     (or (= y_1 0.0) (= y_1 2.0))
     (or (= y_2 0.0) (= y_2 4.0)))
)
