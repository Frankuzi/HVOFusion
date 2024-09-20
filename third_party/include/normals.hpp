#pragma once
#include <cmath>
#include <Eigen/Dense>
#include "octree.h"


namespace Eigen
{

    template <typename Vector, typename Scalar>
    struct EigenVector
    {
        Vector vector;
        Scalar length;
    };

    template <typename Matrix, typename Vector>
    inline void computeEigen33(const Matrix &mat, typename Matrix::Scalar &eigenvalue, Vector &eigenvector);

    template <typename Matrix, typename Vector>
    inline void computeEigen33Values(const Matrix &mat, Vector &eigenvalue);

    template <typename Matrix, typename Roots>
    inline void computeRoots(const Matrix &m, Roots &roots);

    template <typename Scalar, typename Roots>
    inline void computeRoots2(const Scalar &b, const Scalar &c, Roots &roots);

    template <typename Vector, typename Matrix>
    static EigenVector<Vector, typename Matrix::Scalar> getLargest3x3Eigenvector(const Matrix scaledMatrix);

    template <typename Scalar, typename Roots>
    inline void
    computeRoots2(const Scalar &b, const Scalar &c, Roots &roots)
    {
        roots(0) = Scalar(0);
        Scalar d = Scalar(b * b - 4.0 * c);
        if (d < 0.0) // no real roots ! THIS SHOULD NOT HAPPEN!
            d = 0.0;

        Scalar sd = std::sqrt(d);

        roots(2) = 0.5f * (b + sd);
        roots(1) = 0.5f * (b - sd);
    }

    template <typename Matrix, typename Roots>
    inline void
    computeRoots(const Matrix &m, Roots &roots)
    {
        using Scalar = typename Matrix::Scalar;

        // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
        // eigenvalues are the roots to this equation, all guaranteed to be
        // real-valued, because the matrix is symmetric.
        Scalar c0 = m(0, 0) * m(1, 1) *
                        m(2, 2) +
                    Scalar(2) *
                        m(0, 1) * m(0, 2) *
                        m(1, 2) -
                    m(0, 0) *
                        m(1, 2) * m(1, 2) -
                    m(1, 1) * m(0, 2) *
                        m(0, 2) -
                    m(2, 2) *
                        m(0, 1) * m(0, 1);
        Scalar c1 = m(0, 0) * m(1, 1) -
                    m(0, 1) * m(0, 1) +
                    m(0, 0) * m(2, 2) -
                    m(0, 2) * m(0, 2) +
                    m(1, 1) * m(2, 2) -
                    m(1, 2) * m(1, 2);
        Scalar c2 = m(0, 0) + m(1, 1) + m(2, 2);

        if (std::abs(c0) < Eigen::NumTraits<Scalar>::epsilon()) // one root is 0 -> quadratic equation
            computeRoots2(c2, c1, roots);
        else
        {
            constexpr Scalar s_inv3 = Scalar(1.0 / 3.0);
            const Scalar s_sqrt3 = std::sqrt(Scalar(3.0));
            // Construct the parameters used in classifying the roots of the equation
            // and in solving the equation for the roots in closed form.
            Scalar c2_over_3 = c2 * s_inv3;
            Scalar a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
            if (a_over_3 > Scalar(0))
                a_over_3 = Scalar(0);

            Scalar half_b = Scalar(0.5) * (c0 + c2_over_3 * (Scalar(2) * c2_over_3 * c2_over_3 - c1));

            Scalar q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
            if (q > Scalar(0))
                q = Scalar(0);

            // Compute the eigenvalues by solving for the roots of the polynomial.
            Scalar rho = std::sqrt(-a_over_3);
            Scalar theta = std::atan2(std::sqrt(-q), half_b) * s_inv3;
            Scalar cos_theta = std::cos(theta);
            Scalar sin_theta = std::sin(theta);
            roots(0) = c2_over_3 + Scalar(2) * rho * cos_theta;
            roots(1) = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
            roots(2) = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

            // Sort in increasing order.
            if (roots(0) >= roots(1))
                std::swap(roots(0), roots(1));
            if (roots(1) >= roots(2))
            {
                std::swap(roots(1), roots(2));
                if (roots(0) >= roots(1))
                    std::swap(roots(0), roots(1));
            }

            if (roots(0) <= 0) // eigenval for symmetric positive semi-definite matrix can not be negative! Set it to 0
                computeRoots2(c2, c1, roots);
        }
    }

    template <typename Vector, typename Matrix>
    static EigenVector<Vector, typename Matrix::Scalar>
    getLargest3x3Eigenvector(const Matrix scaledMatrix)
    {
        using Scalar = typename Matrix::Scalar;
        using Index = typename Matrix::Index;

        Matrix crossProduct;
        crossProduct << scaledMatrix.row(0).cross(scaledMatrix.row(1)),
            scaledMatrix.row(0).cross(scaledMatrix.row(2)),
            scaledMatrix.row(1).cross(scaledMatrix.row(2));

        // expression template, no evaluation here
        const auto len = crossProduct.rowwise().norm();

        Index index;
        const Scalar length = len.maxCoeff(&index); // <- first evaluation
        return {crossProduct.row(index) / length, length};
    }

    template <typename Matrix, typename Vector>
    inline void computeEigen33(const Matrix &mat, typename Matrix::Scalar &eigenvalue, Vector &eigenvector)
    {
        using Scalar = typename Matrix::Scalar;
        // Scale the matrix so its entries are in [-1,1].  The scaling is applied
        // only when at least one matrix entry has magnitude larger than 1.  

        Scalar scale = mat.cwiseAbs().maxCoeff(); 
        if (scale <= std::numeric_limits<Scalar>::min())
            scale = Scalar(1.0);

        Matrix scaledMat = mat / scale; 

        Vector eigenvalues;
        computeRoots(scaledMat, eigenvalues);

        eigenvalue = eigenvalues(0) * scale;

        scaledMat.diagonal().array() -= eigenvalues(0);

        eigenvector = getLargest3x3Eigenvector<Vector>(scaledMat).vector;
    }

    template <typename Matrix, typename Vector>
    inline void computeEigen33Values(const Matrix &mat, Vector &eigenvalue)
    {
        using Scalar = typename Matrix::Scalar;
        // Scale the matrix so its entries are in [-1,1].  The scaling is applied
        // only when at least one matrix entry has magnitude larger than 1.   

        Scalar scale = mat.cwiseAbs().maxCoeff(); 
        if (scale <= std::numeric_limits<Scalar>::min())
            scale = Scalar(1.0);

        Matrix scaledMat = mat / scale; 

        Vector eigenvalues;
        computeRoots(scaledMat, eigenvalues);

        eigenvalue = eigenvalues * scale;
    }

}
