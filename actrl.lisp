;;; ACT-RL / Reinforcement Learning library for ACT-R
;;; Implements standard RL algorithms in ACT-R

(defparameter *learning-rate*)

(defparameter *q-table* (make-hash-table))

(defparameter *update-queue* ())

(defparameter *temporal-discount* 0.99)

(defun production-name? (name)
  "Checks whether a string names a production"
  t)

(defun seq (start end &optional (step 1))
  "Creates a list with a range of numbers"
  (let ((results nil)
	(partial start))
    (cond ((and (< start end)
		(plusp step))
	   (loop while (< partial end) do
	     (push partial results)
	     (incf partial step)))
	  ((and (> start end)
		(minusp step))
	   (loop while (> partial end) do
	     (push partial results)
	     (incf partial step)))
	  (t
	   nil))
    (reverse results)))

(defun q-list? (lst)
  "Checks whether a list is an assoc list of productions and q-values"
  (and (consp lst)
	   (every #'consp lst)
	   ;;(every #'(lambda (x) (= 2 (length x))) lst)
	   (every #'symbolp (mapcar #'car lst))
	   (every #'numberp (mapcar #'cdr lst))))

(defun needle (cum-probs)
  "Spins a bottle"
  (let* ((spin (random 1.0))
		 (limited (remove-if #'(lambda (x) (> x spin))
							 cum-probs)))
	(length limited)))
  
(defun boltzmann-policy (qlist temp)
  (when (q-list? qlist)
	(let* ((qvals (mapcar #'cdr qlist))
		   (expq (mapcar #'(lambda (x) (exp (/ x temp)))
						 qvals))
		   (sum (apply #'+ expq))
		   (probs (mapcar #'(lambda (x) (/ x sum)) expq))
		   (ii (seq 0 (length qvals)))
		   (cumul (mapcar #'(lambda (i) (reduce #'+ (subseq probs 0 (1+ i))))
						  ii)))
	  (car (nth (needle cumul) qlist))))) 

(defun update-q-value (q reward alpha gamma elapsed-time qnext)
  (+ q
	 (* alpha
		(+ reward (* (expt gamma elapsed-time)
					 qnext)
		   (* -1 q)))))
	 

