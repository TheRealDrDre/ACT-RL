;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Author      : Andrea Stocco
;;;
;;; Address     : Department of Psychology 
;;;             : University of Washington
;;;             : Seattle, WA 98195
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Filename    : actrl.lisp
;;; Version     : 0.1
;;; 
;;; Description : A module that provides ACT-R with basic, classic Rl algorithms.
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; ----- History -----
;;;
;;; 2018-08-10  : * Created and started github code
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; ------------------------------------------------------------------
;;; PARAMETERS
;;; ------------------------------------------------------------------

(defparameter *default-learning-rate* 0.1)

(defparameter *default-temporal-discount* 0.99)

(defparameter *default-default-q-value* 0)

(defparameter *default-on-policy* t)

(defparameter *default-temperature* 0.001)


;;; ------------------------------------------------------------------
;;; UTILITIES
;;; ------------------------------------------------------------------

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

;;; ------------------------------------------------------------------
;;; MODULE AND DATA STRUCTURES
;;; ------------------------------------------------------------------
;;; The module is just a collection of parameters and parametrized
;;; data structures.
;;; ------------------------------------------------------------------

(defstruct rl-module
  ;; Parameters
  temperature learning-rate
  temporal-discount default-q-value
  on-policy depth
 
  ;; Structures
  q-table p-history q-history)


(defun q-list? (lst)
  "Checks whether a list is an assoc list of productions and q-values"
  (and (consp lst)
	   (every #'consp lst)
	   (every #'symbolp (mapcar #'car lst))
	   (every #'numberp (mapcar #'cdr lst))))


(defmethod q-value (prod)
  "Returns the q-value associated with a give production in a Q-module"
  (let ((mod (get-module 'rl)))
	(unless (null mod)
	  (let ((qtable (q-table mod)))
		(unless (member prod (hash-table-keys qtable))
		  (setf (gethash prod qtable)
				(default-q-value mod)))
		(gethash prod qtable)))))


(defmethod set-q-value (prod qval)
  "Sets the q-value associated with a give production in a Q-module"
  (let ((mod (get-module 'rl)))
	(unless (null mod)
	  (let ((qtable (q-table mod)))
		(setf (gethash prod qtable)
			  qval)))))


(defun create-q-list (c-set)
  "Creates an production/q-value assoc list from a conflict set"
  (mapcar #'(lambda (x)
			  (cons x (q-value x)))
		  c-set))


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


;;; This is just to remind myself how to implement a module

(defun create-rl-module (model-name)
  "Creates an RL module (instance of class ACT-RL"
  (declare (ignore model-name))
  (make-rl-module))


(defun delete-rl-module (mod)
  "Deletes the RL module"
  (setf mod nil))


(defun reset-rl-module (mdl)
  "Resets the RL module (Does nothing for now)"
  (when (and (current-model)
			 mdl)
	(let ((def-q (rl-module-default-q-value mdl))
		  (prods (no-output (pp))))
	  (loop for prod in prods do
		   (set-q-value prod def-q)))))


(define-module-fct 'rl '()
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
			:temperature
			:documentation "Temperature T in Boltzmann selection"
			:default-value *default-temperature*
			:valid-test #'(lambda (x)
							(and (numberp x)
								 (plusp x))))
		
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


;;; MODULE-QUERY
;;; --------------------------------------------------------------
;;; A query that handles the basic state checks, i.e. state free,
;;; busy, error.  
;;; --------------------------------------------------------------
(defun rl-module-query (mod buffer slot value)
  "Simple query function for the math module"
  (case slot
    (state
     (case value
       (error nil)
       (busy (math-module-busy mod))
       (free (not (math-module-busy mod)))
       (t (print-warning "Bad state query to ~s buffer" buffer))))
    (t (print-warning 
     "Invalid slot ~s in query to buffer ~s" query buffer))))
