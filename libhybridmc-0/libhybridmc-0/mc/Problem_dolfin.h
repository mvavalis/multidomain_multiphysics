#ifndef __PROBLEM_DOLFIN_H__
#define __PROBLEM_DOLFIN_H__

#include <memory>
#include <dolfin/common/Array.h>
#include <dolfin/function/Expression.h>

template <int dim>
class Problem {
public:
    std::shared_ptr<const dolfin::Expression> f_expr;
    std::shared_ptr<const dolfin::Expression> q_expr;

    double D[dim];

    Problem(const double *D,
            std::shared_ptr<const dolfin::Expression> _f = 0,
            std::shared_ptr<const dolfin::Expression> _q = 0) :
        f_expr(_f), q_expr(_q)
    {
        for (int i=0; i<dim; ++i)
            this->D[i] = D[i];
    }

    double f(const double *_x)
    {
        dolfin::Array<double> values(1);
        dolfin::Array<double> x(dim,const_cast<double *>(_x));
        f_expr->eval(values,x);
        return values[0];
    }

    double q(const double *_x)
    {
        dolfin::Array<double> values(1);
        dolfin::Array<double> x(dim,const_cast<double *>(_x));
        q_expr->eval(values,x);
        return values[0];
    }

    double test_u(const double *_x) { return f(_x); }

};

#endif
