#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION  //Setting the minimum version for Numpy API to 1.7
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>

struct pair {
    double value;
    int index;
} typedef pair_t;

struct roc_point {
    int tp;
    int fp;
    double threshold;
} typedef roc_point_t;

// int compare(const void *a, const void *b);
// roc_point_t *getCurve(PyObject *y_score, int *y_true_int, int classIndex, int *count_vec);
// double getROCArea(roc_point_t *points, int n_instances);

int compare(const void *a, const void *b) {
  return ((pair_t*)a)->value >= ((pair_t*)b)->value;
}

roc_point_t *getCurve(PyObject *y_score, int *y_true_int, int classIndex, int *count_vec) {
    if((y_score == NULL) || (y_true_int == NULL)) {
        return NULL;
    }

    npy_intp *y_score_dims = PyArray_DIMS((PyArrayObject*)y_score);
    int n_instances = (int)y_score_dims[0], n_classes = (int)y_score_dims[1];

    if ((n_instances == 0) || (n_classes <= classIndex)) {
      return NULL;
    }

    npy_intp y_score_itemsize = PyArray_ITEMSIZE((PyArrayObject*)y_score);

    char *y_score_ptr = PyArray_BYTES((PyArrayObject*)y_score);

    pair_t *pairedArray = (pair_t*)malloc(sizeof(pair_t) * n_instances);

    // Weka code starts below

    int totPos = 0, totNeg = 0;

    // Get distribution of positive/negatives
    double prob;
    for (int i = 0; i < n_instances; i++) {
        for(int j = 0; j < n_classes; j++) {
            prob = PyFloat_AsDouble(PyArray_GETITEM((PyArrayObject*)y_score, y_score_ptr));
            y_score_ptr += y_score_itemsize;

            if(j == classIndex) {
                pairedArray[i] = {.value = prob, .index = i};
            }
        }

        if(y_true_int[i] == classIndex) {
            totPos += 1;
        } else {
            totNeg += 1;
        }
    }

    qsort(pairedArray, n_instances, sizeof(pair_t), compare);

    double threshold = 0;
    int cumulativePos = 0, cumulativeNeg = 0;
    int fp = totNeg, fn = 0, tp = totPos, tn = 0;

    roc_point_t *points = (roc_point_t*)malloc(sizeof(roc_point_t) * (n_instances + 1));

    *count_vec = 0;
    for(int i = 0; i < n_instances; i++) {
        if((i == 0) || (pairedArray[i].value > threshold)) {
            tp = tp - cumulativePos;  // true positive
            fn = fn + cumulativePos;  // false negative
            fp = fp - cumulativeNeg;  // false positive
            tn = tn + cumulativeNeg;  // true negative
            threshold = pairedArray[i].value;
            points[*count_vec] = {.tp = tp, .fp = fp, .threshold = threshold};
            *count_vec = *count_vec + 1;
            cumulativeNeg = 0;
            cumulativePos = 0;
            if(i == (n_instances - 1)) {
                break;
            }
        }

        if(y_true_int[pairedArray[i].index] == classIndex) {
            cumulativePos += 1;
        } else {
            cumulativeNeg += 1;
        }
    }

    // make sure a zero point gets into the curve
    // if((fn != totPos) || (tn != totNeg)) {
    tp = 0;
    fp = 0;
    tn = totNeg;
    fn = totPos;
    threshold = pairedArray[n_instances - 1].value + 10e-6;
    points[*count_vec] = {.tp = tp, .fp = fp, .threshold = threshold};

    free(pairedArray);

    return points;
}

double getROCArea(roc_point_t *points, int n_instances) {
    int n = n_instances + 1;
    if(points == NULL) {
        return NAN;
    }

    double area = 0.0, cumNeg = 0.0;
    double totalPos = points[0].tp;
    double totalNeg = points[0].fp;
    for(int i = 0; i < n; i++) {
        double cip, cin;
        if(i < (n - 1)) {
            cip = points[i].tp - points[i + 1].tp;
            cin = points[i].fp - points[i + 1].fp;
        }  else {
            cip = points[n - 1].tp;
            cin = points[n - 1].fp;
        }
        area += cip * (cumNeg + (0.5 * cin));
        cumNeg += cin;
    }
    area /= (totalNeg * totalPos);

    return area;
}

// typedef struct {
//     PyObject_HEAD
//     metricsClass *ptr_autocve;
// } Pymetrics;

/*Object initialization method (expect to receive the pameters of optmization process)*/
static int Pymetrics_init(PyObject *self, PyObject *args, PyObject *kwargs){
//    PyObject *timeout_pip_sec=NULL;
//    PyObject *scoring=NULL;
//
//    int seed=42;
//    int timeout_evolution_process_sec=0;
//    int n_jobs=1;
//    int size_pop_components=50, size_pop_ensemble=50, generations=100;
//    int verbose=0;
//    char *grammar_file="grammarTPOT";
//    double elite_portion_components=0.1, mut_rate_components=0.9, cross_rate_components=0.9;
//    double elite_portion_ensemble=0.1, mut_rate_ensemble=0.1, cross_rate_ensemble=0.9;
//    int cv_folds=5;
//
//    static char *keywords[]={"random_state","n_jobs","max_pipeline_time_secs","max_evolution_time_secs","grammar","generations","population_size_components","mutation_rate_components","crossover_rate_components","population_size_ensemble","mutation_rate_ensemble","crossover_rate_ensemble","scoring","cv_folds","verbose", NULL}; //NULL-terminated array
//
//    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"|$iiOisiiddiddOii",keywords, &seed, &n_jobs, &timeout_pip_sec, &timeout_evolution_process_sec, &grammar_file, &generations, &size_pop_components, &mut_rate_components, &cross_rate_components, &size_pop_ensemble, &mut_rate_ensemble, &cross_rate_ensemble, &scoring, &cv_folds, &verbose)) //Function and arguments |$ before keyword args
//        return NULL;
//
//    if(timeout_pip_sec==NULL)
//        timeout_pip_sec=Py_BuildValue("i", 60);
//    else if(timeout_pip_sec==Py_None || (PyLong_Check(timeout_pip_sec) && PyLong_AsLong(timeout_pip_sec)>0))
//        Py_XINCREF(timeout_pip_sec);
//    else{
//        PyErr_SetString(PyExc_TypeError, "max_pipeline_time_secs must be an integer greater than zero or None");
//        return NULL;
//    }
//
//    if(scoring==NULL)
//        scoring=Py_BuildValue("s","balanced_accuracy");
//    else
//        Py_XINCREF(scoring);
//
//    if(self->ptr_autocve)
//        delete self->ptr_autocve;
//
//    try{
//        self->ptr_autocve=new metricsClass(seed, n_jobs, timeout_pip_sec, timeout_evolution_process_sec, grammar_file, generations, size_pop_components, elite_portion_components, mut_rate_components, cross_rate_components, size_pop_ensemble, elite_portion_ensemble, mut_rate_ensemble, cross_rate_ensemble, scoring, cv_folds, verbose);
//    }catch(const char *e){
//        PyErr_SetString(PyExc_Exception, e);
//        return NULL;
//    }


    return 1;
}

static PyObject *Pymetrics_get_unweighted_area_under_roc(PyObject *self, PyObject *args, PyObject *kwargs) {

    static char *kwds[] = {"y_true", "y_score", NULL};
    PyObject *y_true, *y_score;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO", kwds, &y_true, &y_score)) {
        return NULL;
    }

    if(PyArray_NDIM((PyArrayObject*)y_score) > 2) {
        PyErr_SetString(PyExc_ValueError, "Probability matrix should have two dimensions.");
        return NULL;
    }

    y_true = (PyObject*)PyArray_GETCONTIGUOUS((PyArrayObject*)y_true);
    y_score = (PyObject*)PyArray_GETCONTIGUOUS((PyArrayObject*)y_score);

    npy_intp *y_pred_dims = PyArray_DIMS((PyArrayObject*)y_score);
    int n_instances = (int)y_pred_dims[0], n_classes = (int)y_pred_dims[1];

    roc_point_t *curve;

    double area, mean_area = 0.0;

    char *y_true_ptr = PyArray_BYTES((PyArrayObject*)y_true);
    int *y_true_int = (int*)malloc(sizeof(int) * n_instances);
    npy_intp y_true_itemsize = PyArray_ITEMSIZE((PyArrayObject*)y_true);

    for(int i = 0; i < n_instances; i++) {
        y_true_int[i] = PyLong_AsLong(PyArray_GETITEM((PyArrayObject*)y_true, y_true_ptr));
        y_true_ptr += y_true_itemsize;
    }

    int count_vec;
    for(int c = 0; c < n_classes; c++) {
        curve = getCurve(y_score, y_true_int, c, &count_vec);
        area = getROCArea(curve, count_vec);
        mean_area += area;
        free(curve);
    }

    free(y_true_int);

    return Py_BuildValue("d", mean_area / n_classes);
}


static PyMethodDef metrics_methods[] = {
    { "get_unweighted_area_under_roc", (PyCFunction)Pymetrics_get_unweighted_area_under_roc, METH_VARARGS | METH_KEYWORDS, "Get unweighted area under the ROC curve for a set of predictions." },
    {NULL}  /* Sentinel */
};

static PyTypeObject PymetricsType = {PyVarObject_HEAD_INIT(NULL, 0)
                                    "metricsClassifier"   /* tp_name */
};


//#if PY_MAJOR_VERSION >= 3

static PyModuleDef metrics_module = {
    PyModuleDef_HEAD_INIT,
    "metrics",   /* name of module */
    "Module with metrics used in the evolutionary process of PBIL.", /* module documentation, may be NULL */
    -1,//sizeof(struct module_state),       /* size of per-interpreter state of the module,or -1 if the module keeps state in global variables. */
    metrics_methods
};


PyMODINIT_FUNC PyInit_metrics(void){
//    PyObject* module_py;
//
//    PymetricsType.tp_new = PyType_GenericNew;
//    PymetricsType.tp_basicsize=sizeof(Pymetrics);
//    PymetricsType.tp_dealloc=(destructor) Pymetrics_dealloc;
//    PymetricsType.tp_flags=Py_TPFLAGS_DEFAULT;
//    PymetricsType.tp_doc="metrics Classifier";
//    PymetricsType.tp_methods=Pymetrics_methods;
//    //~ PymetricsType.tp_members=Noddy_members;
//    PymetricsType.tp_init=(initproc)Pymetrics_init;
//
//    if (PyType_Ready(&PymetricsType) < 0)
//        return NULL;
//
//    module_py = PyModule_Create(&metrics_module);
//    if (!module_py)
//        return NULL;
//
//    Py_INCREF(&PymetricsType);
//    PyModule_AddObject(module_py, "metricsClassifier", (PyObject *)&PymetricsType);
//    return module_py;
    Py_Initialize();
    import_array();  // import numpy arrays
    return PyModule_Create(&metrics_module);
}
