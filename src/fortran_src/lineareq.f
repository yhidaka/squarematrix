      subroutine lineareq_block(an,vn,dxn,bn,n,k)
cf2py intent(in) :: an,vn
cf2py intent(out) :: bn,dxn
ccf2py intent(in) :: a
cf2py intent(hide) :: n,L,U,b,d,x,i,j,k
      integer :: n,k
      double precision :: an(k, n, n),am(n,n), bn(k,n,n)
      double precision :: a(n, n)
      double precision :: L(n,n),U(n,n),b(n),d(n),x(n),vn(k,n),dxn(k,n)
      double precision coeff
      L=0.0
      U=0.0
      b=0.0
      d=0.0
      x=0.0
      dxn=0.0
      coeff=0.0

      do ik=1,k

              a=an(ik,:,:)
c step 1: forward elimination
              do k=1,n-1
                do i=k+1,n
                    coeff=a(i,k)/a(k,k)
                    L(i,k) = coeff
                    do j=k+1,n
                          a(i,j) = a(i,j)-coeff*a(k,j)
                          am(i,j)=0.0
                    end do
                end do
              end do
c Step 2: prepare L and U matrices 
c L matrix is a matrix of the elimination coefficient
c + the diagonal elements are 1.0
              do i=1,n
                      L(i,i) = 1.0
              end do

c U matrix is the upper triangular part of A
              do j=1,n
                  do i=1,j
                    U(i,j) = a(i,j)
                  end do
              end do


c Step 3: compute columns of the inverse matrix C
              do k=1,n
                  b(k)=1.0
                  d(1) = b(1)
c Step 3a: Solve Ld=b using the forward substitution
                  do i=2,n
                    d(i)=b(i)
                    do j=1,i-1
                      d(i) = d(i) - L(i,j)*d(j)
                    end do
                  end do
c Step 3b: Solve Ux=d using the back substitution
                  x(n)=d(n)/U(n,n)
                  do i = n-1,1,-1
                    x(i) = d(i)
                    do j=n,i+1,-1
                      x(i)=x(i)-U(i,j)*x(j)
                    end do
                    x(i) = x(i)/u(i,i)
                  end do
c Step 3c: fill the solutions x(n) into column k of C
                  do i=1,n
                    am(i,k) = x(i)
                  end do
                  b(k)=0.0
              end do
              bn(ik,:,:)=am
c              print *,'ik, am(1,1),am(1,2)=', ik,am(1,1),am(1,2)
c              print *,'ik, am(2,1),am(2,2)=', ik,am(2,1),am(2,2)

              do i=1,n
                  dxn(ik,i)=0
                  do j=1,n
                   dxn(ik,i)=dxn(ik,i)-am(i,j)*vn(ik,j)
                 end do
              end do
      end do
      end subroutine lineareq_block