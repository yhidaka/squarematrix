	subroutine zcol(Zxs,nod,powerindex,m,n,zout)
cf2py intent(in) :: Zxs
cf2py intent(in) :: nod
cf2py intent(in) :: powerindex
cf2py intent(hide) :: m,n
cf2py intent(out) :: zout
	    double complex :: Zxs(n)
	    integer :: nod, n,m
	    integer :: powerindex(m,n)
	    double complex :: zout(m)

c First start from [1,x], then build [1,x,x^2], then iterate to "norder".
	    double complex :: u(n,nod+1)
	    integer :: nz(n)
	    integer :: pinx_i(n)
	    double complex :: row
	    integer :: p
          do 100 i=1,n
              u(i,1)=1
100       continue

	    do 200 j=2,nod+1
	        do 300 i=1,n
		      u(i,j)=Zxs(i)*u(i,j-1)
300	        continue
200	    continue

c Use the table built above, i.e., zxn,zxnp,.. to construct the square matrix.
	    
	    do 400 i=1,m
		  do 600 j=1,n
		  pinx_i(j) = powerindex(i,j)
600		  continue
	        call findnonzeros(pinx_i,n,nz,ln)
	        if (ln==0) then
		      row = 1.0
		  else
		      j = nz(1)
		      k = powerindex(i,j)
		      row = u(j,k+1)
			if (ln>1) then
   		          do 500 p=2,ln
			        j = nz(p)
			        k = powerindex(i,j)
			        row = row*u(j,k+1)
500			    continue
			else
			end if 
		  end if
		  zout(i) = row      
400	    continue
	end subroutine zcol

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
100	    continue
	end subroutine findnonzeros
