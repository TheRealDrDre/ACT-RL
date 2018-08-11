;;; ACT-RL / Reinforcement Learning library for ACT-R
;;; Implements standard RL algorithms in ACT-R

(defparameter *default-learning-rate* 0.1)

(defparameter *default-temporal-discount* 0.99)

(defparameter *default-default-q-value* 0)

(defparameter *default-on-policy* t) 

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


(defun needle (cum-probs)
  "Spins a bottle"
  (let* ((spin (random 1.0))
		 (limited (remove-if #'(lambda (x) (> x spin))
							 cum-probs)))
	(length limited)))

;;; Data structures

(defun q-list? (lst)
  "Checks whether a list is an assoc list of productions and q-values"
  (and (consp lst)
	   (every #'consp lst)
	   (every #'symbolp (mapcar #'car lst))
	   (every #'numberp (mapcar #'cdr lst))))

(defclass act-rl-module ()
  ((q-table :accessor q-table
			:initform (make-hash-table))
   (p-history :accessor p-history
			  :initform nil)
   (q-history :accessor q-history
			  :initform nil))
  (:documentation "An ACT-R module that replaces utility with classic Q-values "))


(defmethod q-value (prod module)
  "Returns the q-value associated with a give production in a Q-module"
  (unless (null module)
	(let ((qtable (q-table module)))
	  (unless (member prod (hash-table-keys qtable))
		(let ((
		(setf (gethash prod qtable) 
		  

(defun create-q-list (cset)
  "Creates an production/q-value assoc list from a conflict set"
  (mapcar #'(lambda (x)
			  (cons x (q-value x)))
		  cset))

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
  "Calculates the new Q-value, given the classical arguments plus the elapsed time"
  (+ q
	 (* alpha
		(+ reward (* (expt gamma elapsed-time)
					 qnext)
		   (* -1 q)))))


(define-module-fct 'act-rl '()
  (list (define-parameter
			:learning-rate
			:documentation "Learning rate alpha"
			:default-value *default-learning-rate*
			:valid-test #'(lambda (alpha)
							(and (numberp alpha)
								 (>= alpha 0)))
			:warning "Non-negative number"
			:owner t)
		(define-parameter
			:temporal-discount
			:documentation "Temporal discount gamma"
			:default-value *default-temporal-discounting*
			:valid-test #'(lambda (gamma)
							(and (numberp gamma)
								 (> gamma 0)
								 (<= gamma 1))))
		(define-parameter
			:temporal-discount
			:documentation "Temporal discount gamma"
			:default-value *default-temporal-discounting*
			:valid-test #'(lambda (gamma)
							(and (numberp gamma)
								 (> gamma 0)
								 (<= gamma 1))))
		(define-parameter
			:default-q-value
			:documentation "Initial Q-value of a production"
			:default-value *default-default-q-value*
			:valid-test #'numberp)

		(define-parameter
			:on-policy
			:documentation "Whether on-policy (SARSA) or off-policy (Q-learning)"
			:default-value *default-on-policy*
			:valid-test #'(lambda (x) (member x '(t nil))))
		
		(define-parameter :esc :owner nil))
  :version "1.0a1"
  :documentation "First version of RL module")
