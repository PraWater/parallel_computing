#include <math.h>
#include <stdio.h>
int main() {
  float a[10][10], b[10], x[10], xn[10], sum, e;
  int i, j, n, flag = 0, key;
  printf("\nThis program illustrates Gauss-Jacobi method to solve system of "
         "AX=B\n");
  printf("\nEnter the dimensions of coefficient matrix n: ");
  scanf("%d", &n);
  printf("\nEnter the elements of matrix A:\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      scanf("%f", &a[i][j]);
    }
  }
  printf("\nEnter the elements of matrix B:\n");
  for (i = 0; i < n; i++)
    scanf("%f", &b[i]);
  printf("\nThe system of linear equations:\n");
  for (i = 0; i < n; i++) {
    printf("\n(%.2f)x1+(%.2f)x2+(%.2f)x3=(%.2f)\n", a[i][0], a[i][1], a[i][2],
           b[i]);
  }
  for (i = 0; i < n; i++) {
    sum = 0;
    for (j = 0; j < n; j++) {
      sum += fabs(a[i][j]);
    }
    sum -= fabs(a[i][i]);
    if (fabs(a[i][i]) < sum) {
      flag = 1;
      break;
    }
  }
  if (flag == 1) {
    printf("\nThe system of linear equations are not diagonally dominant\n");
    return main();
  } else {
    printf("\nThe system of linear equations are diagonally dominant\n");
    printf("\nEnter the initial approximations: ");
    for (i = 0; i < n; i++) {
      // x[i]=0;
      printf("\nx%d=", (i + 1));
      scanf("%f", &x[i]);
    }
    printf("\nEnter the error tolerance level:\n");
    scanf("%f", &e);
  }
  printf("x[1]\t\tx[2]\t\tx[3]");
  printf("\n");
  key = 0;
  while (key < n - 1) {
    key = 0;
    for (i = 0; i < n; i++) {
      sum = b[i];
      for (j = 0; j < n; j++)
        if (j != i)
          sum -= a[i][j] * x[j];
      xn[i] = sum / a[i][i];
      if (fabs(x[i] - xn[i]) < e) {
        key = key + 1;
      }
    }
    if (key == n) {
      break;
    }
    printf("%f\t %f\t %f\t", xn[0], xn[1], xn[2]);
    for (i = 0; i < n; i++) {
      x[i] = xn[i];
    }
  }
  printf("\nAn approximate solution to the given system of equations is\n");
  for (i = 0; i < n; i++) {
    printf("\nx[%d]=%f\n", (i + 1), x[i]);
  }
  return 0;
}
