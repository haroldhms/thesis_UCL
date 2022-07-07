==============================================================================================

This code implements the algorithm described in the paper:
      "Nonparametric Canonical Correlation Analysis"
     by Tomer Michaeli, Weiran Wang and Karen Livescu
 International Conference on Machine Learning (ICML 2016)

==============================================================================================
Quick Start
==============================================================================================

A demonstration script is included: NCCA_ToyExample1.m.

==============================================================================================
Contents
==============================================================================================

The package comprises these functions

*) NCCA.m: 
   - Description: Nonparametric canonical correlation analysis 
   - Inputs:	X,Y - paired training examples (rows) of features (columns).
		d - dimension of the output transformed features.
		X_unpaired, Y_unpaired [optional] - additional unpaired examples. The
 numbers 
			of unpaired X and Y examples does not need to be identical.
		params [optional] - structure with algorithm paramters:
			- hx,hy - bandwidth parameters for the KDEs of views 1,2. Default
				is  0.5.
			- nnx,nny - number of nearest neighbors for the KDEs of views 1,2.

				Default is 20.
			- randSVDiters - number of iterations for random SVD algorithm. Set
				 higher for better accuracy, default is 20.
			- randSVDblock - block size for random SVD algorithm. Set higher for
				better accuracy, default is d+10;
			- doublyStIters - number of iterations for doubly stichastic 
				normalization. Set higher for better accuracy, default is 15.
   
   - Outputs: 	Xproj,Yproj - d-dimensional projections of the training data.
		cor - correlations between the d pairs of projections.
		XunpairedProj,YunpairedProj [optional] - d-dimensional projections of the
			unpaired data.
		OSEoutputs [optional] - outputs needed for out-of-sample extension.


*) NCCA_OSE_view1.m
   - Description: Computes the Nystrom out-of-sample extension for additional view 1 examples
   - Inputs:	Xnew - new view 1 examples (rows) of features (columns).
		Xtr - original training examples of view 1 which were used to obtain the 
			projections with NCCA.
		Yproj - the projections of the view 2 examples which were obtained with
			NCCA.
		cor - the correlations between the projections which were obtained with
			NCCA.
		OSEoutputs - output structure from the NCCA.m function, needed for
			out-of-sample extension.
   
   - Outputs:   XnewProj - d-dimensional projections of the new view 1 examples.


*) NCCA_OSE_view2.m
   - Description: Computes the Nystrom out-of-sample extension for additional view 2 examples
   - Inputs:	Ynew - new view 2 examples (rows) of features (columns).
		Ytr - original training examples of view 2 which were used to obtain the 
			projections with NCCA.
		Xproj - the projections of the view 1 examples which were obtained with
			NCCA.
		cor - the correlations between the projections which were obtained with
			NCCA.
		OSEoutputs - output structure from the NCCA.m function, needed for
			out-of-sample extension.
   
   - Outputs:   YnewProj - d-dimensional projections of the new view 2 examples.



==============================================================================================
Feedback
==============================================================================================

If you have any comment, suggestion, or question, please do contact
Tomer: tomer.m at ee.technion.ac.il

==============================================================================================
Citation
==============================================================================================
   
If you use this code in a scientific project, you should cite the following paper in any
resulting publication:

Tomer Michaeli, Weiran Wang and Karen Livescu,
"Nonparametric Canonical Correlation Analysis",
International Conference on Machine Learning (ICML 2016).
==============================================================================================