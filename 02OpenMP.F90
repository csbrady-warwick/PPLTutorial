MODULE sims
    USE iso_fortran_env
    IMPLICIT NONE


    TYPE dataPack
      REAL(REAL64), DIMENSION(:,:), ALLOCATABLE :: U, Uprime
    END TYPE dataPack

    CONTAINS

    SUBROUTINE allocateDataPack(dp,lboundr, uboundr, lboundc, uboundc)
      TYPE(dataPack), INTENT(OUT) :: dp
      INTEGER, INTENT(IN) :: lboundr, uboundr, lboundc, uboundc

      ALLOCATE(dp%U(lboundr:uboundr, lboundc:uboundc))
      ALLOCATE(dp%Uprime(lboundr:uboundr, lboundc:uboundc))
    END SUBROUTINE allocateDataPack

    SUBROUTINE solveMandelBrot(dp)
        TYPE(dataPack), INTENT(INOUT) :: dp
        INTEGER :: ix, iy
        INTEGER :: maxIter
        REAL(REAL64) :: x0, y0, x, y, xtemp
        INTEGER :: iteration

        !$OMP PARALLEL DO PRIVATE(ix, iy, x0, y0, x, y, xtemp, iteration, maxIter) SCHEDULE(dynamic)
        DO iy = LBOUND(dp%U,2), UBOUND(dp%u,2)
          DO ix = LBOUND(dp%U, 1), UBOUND(dp%U, 1)
            x0 = REAL(ix, REAL64) / REAL(SIZE(dp%U, 1), REAL64) * 3.0_REAL64 - 2.0_REAL64
            y0 = REAL(iy, REAL64) / REAL(SIZE(dp%U, 2), REAL64) * 3.0_REAL64 - 1.5_REAL64
            x = 0.0_REAL64
            y = 0.0_REAL64
            iteration = 0
            maxIter = 1000

            DO WHILE (x**2 + y**2 <= 4.0_REAL64 .AND. iteration < maxIter)
              xtemp = x*x - y*y + x0
              y = 2.0_REAL64*x*y + y0
              x = xtemp
              iteration = iteration + 1
            END DO

            dp%U(ix, iy) = REAL(iteration, REAL64)
          END DO
        END DO
    END SUBROUTINE solveMandelBrot

    SUBROUTINE jacobiIterate(dp, numIters)
        TYPE(dataPack), INTENT(INOUT) :: dp
        INTEGER, INTENT(IN) :: numIters
        INTEGER :: iter, i, j
        INTEGER :: nrows, ncols

        nrows = SIZE(dp%U, 1)
        ncols = SIZE(dp%U, 2)

        !$OMP PARALLEL PRIVATE(i, j, iter)
        DO iter = 1, numIters
          !$OMP DO SCHEDULE(static)
          DO j = LBOUND(dp%U, 2)+1, UBOUND(dp%U, 2)-1
            DO i = LBOUND(dp%U, 1)+1, UBOUND(dp%U, 1)-1
              dp%Uprime(i,j) = 0.25_REAL64 * (dp%U(i+1,j) + dp%U(i-1,j) + &
                                              dp%U(i,j+1) + dp%U(i,j-1))
            END DO
          END DO
          !$OMP END DO
          !$OMP DO SCHEDULE(static)
          DO j = LBOUND(dp%U, 2)+1, UBOUND(dp%U, 2)-1
            DO i = LBOUND(dp%U, 1)+1, UBOUND(dp%U, 1)-1
              dp%U(i,j) = dp%Uprime(i,j)
            END DO
          END DO
          !$OMP END DO
          !$OMP BARRIER
        END DO
        !$OMP END PARALLEL
    END SUBROUTINE jacobiIterate

END MODULE sims

PROGRAM main
    USE sims
    IMPLICIT NONE

    TYPE(dataPack) :: dp
    INTEGER :: lboundr, uboundr, lboundc, uboundc
    INTEGER :: numIters, count1, count2, count_rate

    lboundr = 1
    uboundr = 4000
    lboundc = 1
    uboundc = 4000

    numiters=300

    CALL allocateDataPack(dp, lboundr, uboundr, lboundc, uboundc)
    CALL SYSTEM_CLOCK(count1, count_rate)
    CALL solveMandelBrot(dp)
    CALL SYSTEM_CLOCK(count2)
    PRINT *, "Mandelbrot computation time (seconds): ", REAL(count2 - count1, REAL64) / REAL(count_rate, REAL64)

    lboundr = lboundr -1
    uboundr = uboundr +1
    lboundc = lboundc -1
    uboundc = uboundc +1
    CALL allocateDataPack(dp, lboundr, uboundr, lboundc, uboundc)
    CALL SYSTEM_CLOCK(count1)
    CALL jacobiIterate(dp, numIters)
    CALL SYSTEM_CLOCK(count2)
    PRINT *, "Jacobi iteration time (seconds): ", REAL(count2 - count1, REAL64) / REAL(count_rate, REAL64)

END PROGRAM main