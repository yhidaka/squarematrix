      subroutine zcolarray(xs,nod,powerindex,m,n,nturn,zout)
cf2py intent(in) :: xs
cf2py intent(in) :: nod
cf2py intent(in) :: powerindex
cf2py intent(hide) :: m,n,nturn
cf2py intent(out) :: zout
      double complex :: xs(nturn,n)
      integer :: nod, n,m,nturn
      integer :: powerindex(m,n)
      double complex :: zout(nturn,m)
c First start from [1,x], then build [1,x,x^2], then iterate to "norder".
      double complex :: u(nturn,n,nod+1)
      integer :: nz(n)
      integer :: pinx_i(n)
      integer :: p

      u(:,:,1)=1
      do 200 j=2,nod+1
         u(:,:,j)=xs(:,:)*u(:,:,j-1)
200      continue

c Use the table built above, i.e., zxn,zxnp,.. to construct the square matrix.
      
      do 400 i=1,m
      pinx_i = powerindex(i,:)
      call findnonzeros(pinx_i,n,nz,ln)
            if (ln==0) then
                zout(:,i) = 1.0
            else
                    j = nz(1)
                    k = powerindex(i,j)
                    zout(:,i) = u(:,j,k+1)
                    if (ln>1) then
                              do 500 p=2,ln
                                    j = nz(p)
                                    k = powerindex(i,j)
                                    zout(:,i)= zout(:,i)*u(:,j,k+1)
500                           continue
                    else
                    end if 
            end if
400   continue


      end subroutine zcolarray

ccccccccccccccccccccccc
      subroutine findnonzeros(inmp,n,outmp,l)
      integer :: inmp(n)
      integer :: n
      integer :: outmp(n)
      double precision :: tol
      integer :: l
      tol = 1e-12
      l = 0
      do 100 i=1,n
              if (abs(inmp(i))>tol) then
                  l = l+1
                  outmp(l) = i
              else
              endif
100   continue
      end subroutine findnonzeros


