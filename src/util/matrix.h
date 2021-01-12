#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>

// Check if the given floating point array contains a NaN value.
template <typename Derived>
inline bool IsNaN(const Eigen::MatrixBase<Derived>& x);

// Check if the given floating point array contains infinity.
template <typename Derived>
inline bool IsInf(const Eigen::MatrixBase<Derived>& x);

// Perform RQ decomposition on matrix. The RQ decomposition transforms a matrix
// A into the product of an upper triangular matrix R (also known as
// right-triangular) and an orthogonal matrix Q.
template <typename MatrixType>
void DecomposeMatrixRQ(const MatrixType& A, MatrixType* R, MatrixType* Q);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
bool IsNaN(const Eigen::MatrixBase<Derived>& x) {
  return !(x.array() == x.array()).all();
}

template <typename Derived>
bool IsInf(const Eigen::MatrixBase<Derived>& x) {
  return !((x - x).array() == (x - x).array()).all();
}

template <typename MatrixType>
void DecomposeMatrixRQ(const MatrixType& A, MatrixType* R, MatrixType* Q) {
  const MatrixType A_flipud_transpose =
      A.transpose().rowwise().reverse().eval();

  const Eigen::HouseholderQR<MatrixType> QR(A_flipud_transpose);
  const MatrixType& Q0 = QR.householderQ();
  const MatrixType& R0 = QR.matrixQR();

  *R = R0.transpose().colwise().reverse().eval();
  *R = R->rowwise().reverse().eval();
  for (int i = 0; i < R->rows(); ++i) {
    for (int j = 0; j < R->cols() && (R->cols() - j) > (R->rows() - i); ++j) {
      (*R)(i, j) = 0;
    }
  }

  *Q = Q0.transpose().colwise().reverse().eval();

  // Make the decomposition unique by requiring that det(Q) > 0.
  if (Q->determinant() < 0) {
    Q->row(1) *= -1.0;
    R->col(1) *= -1.0;
  }
}