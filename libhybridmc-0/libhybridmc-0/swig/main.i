// module name is set from the build system

%{
#include <numpy/arrayobject.h>
#include <main.h>
#include <dolfin/function/Expression.h>

%}

%init%{
import_array();
%}

%include typemaps/primitives.i
%include typemaps/std_pair.i
%include typemaps/numpy.i
%include typemaps/array.i
%include typemaps/std_vector.i

%include <exception.i>
%include typemaps/exceptions.i

%include <std_shared_ptr.i>
%shared_ptr(dolfin::Expression)

%include <std_string.i>

%include <main.h>
