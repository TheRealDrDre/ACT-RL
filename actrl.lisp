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
	   (every #'symbolp (mapcar #'car lst))
	   (every #'numberp (mapcar #'cdr lst))))

(defun needle (cum-probs)
  "Spins a bottle"
  (let* ((spin (random 1.0))
		 (limited (remove-if #'(lambda (x) (> x spin))
							 cum-probs)))
	(length limited)))
  
(defun boltzmann-policy (qlist temp)
  "Chooses a production with probability proportional to its Q-value" 
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

(defun calculate-q-value (q reward alpha gamma elapsed-time qnext)
  "Calculates the new Q-value, given the classical arguments plys the elapsed time"
  (+ q
	 (* alpha
		(+ reward (* (expt gamma elapsed-time)
					 qnext)
		   (* -1 q)))))
	 

(define-module-fct 'act-rl '()
  (list (define-parameter
			:learning-rate
			:documentation "Learning rate alpha"
			:default-value .1
			:valid-test #'(lambda (alpha)
							(and (numberp alpha)
								 (>= alpha 0)))
			:warning "Non-negative number"
			:owner t)
		(define-parameter
			:temporal-discount
			:documentation "Temporal discount gamma"
			:default-value 0.99
			:valid-test #'(lambda (gamma)
							(and (numberp gamma)
								 (> gamma 0)
								 (<= gamma 1))))
		(define-parameter
			:depth
			:documentation "Number of backpropagation steps"
			:default-value 1  ;; Standard Q-Learning / SARSA
			:valid-test #'(lambda (n)
							(and (integerp n)
								 (plusp n))))
		(define-parameter :esc :owner nil))
  :version "1.0a1"
  :documentation "First version of RL module")
