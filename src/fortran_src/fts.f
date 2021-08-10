c File fts.f
	subroutine fts(pt,ux,uy,bKi4b4,sx,sy,powerindex,nod,l,m,n,
     $   ab0x, ab0y, phib0x, phib0y, dmux, dmuy, ax2, ay2)
	    implicit none
cf2py intent(in) :: pt
cf2py intent(in) :: ux,uy
cf2py intent(in) :: bKi4b4
cf2py intent(in) :: sx,sy
cf2py intent(in) :: powerindex
cf2py intent(in) :: nod
cf2py intent(hide) :: l,m,n
cf2py intent(out) :: ab0x, ab0y, phib0x, phib0y, dmux, dmuy, ax2, ay2

	    double precision :: pt(n)
          double complex :: ux(l,m),uy(l,m)
          double complex :: bKi4b4(n, n)
	    double precision :: sx,sy
	    integer :: powerindex(m, n)
	    integer :: nod
          integer :: l,m,n 
	    
	    double complex :: zbar(n)
	    double complex :: zbarx(n)
	    double complex :: zbary(n)
	    double complex :: Zxs(m)
	    double complex :: Zys(m)
	    double complex :: wx(l)
	    double complex :: wy(l)
	    double complex :: j
          integer :: i
	    double precision :: ab0x,ab0y,phib0x,phib0y,ax2,ay2
	    double complex :: dmux,dmuy
	    call dot_1(bKi4b4,pt,zbar,n,n)
	    zbarx = zbar/sx
	    call zcol(zbarx,nod,powerindex,m,n,Zxs)
          zbary = zbar/sy
	    call zcol(zbary,nod,powerindex,m,n,Zys)
	    call dot_2(ux,Zxs,wx,l,m)
	    call dot_2(uy,Zys,wy,l,m)
	    j = (0,1)
	    dmux = -j*(wx(2))/wx(1)
	    ax2 = abs(wx(3)/wx(1)-(wx(2)/wx(1))**2)
	    ab0x = abs(wx(1))
	    call cphase(wx(1),phib0x)
	    dmuy = -j*(wy(2))/wy(1)
	    ay2 = abs(wy(3)/wy(1)-(wy(2)/wy(1))**2)
	    ab0y = abs(wy(1))
	    call cphase(wy(1),phib0y)
	end subroutine fts

	subroutine zcol(Zxs,nod,powerindex,m,n,
     $   zout)
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

	subroutine dot_1(ff,f,u,l,m)
          double precision :: f(m)
          double complex :: u(l)
          integer :: l,m
          double complex :: ff(l, m)

          do 400 i=1,l
              u(i)=0
400       continue

          do 100 i=1,l
	        do 200 j=1,m
	            u(i)=u(i)+ff(i,j)*f(j)
200	        continue
100       continue
	end subroutine dot_1

	subroutine dot_2(ff,f,u,l,m)
          double complex :: f(m)
          double complex :: u(l)
          integer :: l,m
          double complex :: ff(l, m)

          do 400 i=1,l
              u(i)=0
400       continue

          do 100 i=1,l
	        do 200 j=1,m
	            u(i)=u(i)+ff(i,j)*f(j)
200	        continue
100       continue
	end subroutine dot_2

	subroutine cphase(x,phs)
          double complex :: x
	    double precision:: a,b
          double precision :: phs
	    a = dble(x)
	    b = dimag(x)
	    phs = atan2(b,a)
	end subroutine cphase
